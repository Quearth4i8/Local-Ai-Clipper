"""FastAPI server + local web UI.

Binds to 127.0.0.1 by default: nothing leaves the machine.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import APP_NAME, __version__
from ..cache.store import CacheStore
from ..config import Config, load_config
from ..export.exporters import (_clip_words, export_clips, to_csv, to_json, to_text,
                                write_all, write_ass_for_clip)
from ..llm.base import get_backend
from ..models import fmt_ts
from ..pipeline import JobManager, Pipeline, run_job
from ..transcription.cuda_setup import cuda_available
from ..video.ffmpeg_tools import FFmpeg, FFmpegError, is_media_file
from .filedialog import pick_file

log = logging.getLogger("clipfinder.server")

STATIC_DIR = Path(__file__).parent / "static"
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI(title=APP_NAME, version=__version__, docs_url=None, redoc_url=None)
jobs = JobManager()
CFG: Config = load_config()


def ffmpeg() -> FFmpeg:
    return FFmpeg(CFG.get("general.ffmpeg_path", ""), CFG.get("general.ffprobe_path", ""))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PathBody(BaseModel):
    path: str


class AnalyzeBody(BaseModel):
    path: str
    force_transcribe: bool = False


class ExportBody(BaseModel):
    job_id: str
    formats: List[str] = ["json", "csv", "txt"]


class ExportClipsBody(BaseModel):
    job_id: str
    ranks: Optional[List[int]] = None
    mode: Optional[str] = None
    captions: Optional[bool] = None


class CaptionPreviewBody(BaseModel):
    job_id: str
    rank: int = 1
    seconds: float = 6.0


class ConfigBody(BaseModel):
    config: Dict[str, Any]
    save: bool = True


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = STATIC_DIR / "index.html"
    if not html.exists():
        return HTMLResponse("<h1>UI files are missing (app/ui/static/index.html)</h1>",
                            status_code=500)
    return HTMLResponse(html.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Health / environment
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    ff = ffmpeg()
    ff_ok = ff.available()
    try:
        llm = get_backend(CFG.section("llm")).health()
    except Exception as exc:  # noqa: BLE001
        llm = {"ok": False, "error": str(exc), "models": []}
    return {
        "app": APP_NAME,
        "version": __version__,
        "ffmpeg": {"ok": ff_ok, "path": ff.ffmpeg,
                   "error": None if ff_ok else
                   "FFmpeg not found. Run install.bat or: winget install Gyan.FFmpeg"},
        "cuda": {"ok": cuda_available()},
        "llm": llm,
        "whisper": {"model": CFG.get("whisper.model"), "device": CFG.get("whisper.device")},
        "config": CFG.to_dict(),
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    return CFG.to_dict()


@app.post("/api/config")
def set_config(body: ConfigBody) -> Dict[str, Any]:
    CFG.update(body.config)
    if body.save:
        CFG.save()
    return {"ok": True, "config": CFG.to_dict()}


# ---------------------------------------------------------------------------
# Video selection
# ---------------------------------------------------------------------------
@app.post("/api/browse")
def browse() -> Dict[str, Any]:
    try:
        path = pick_file()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Could not open the file picker: {exc}")
    if not path:
        return {"cancelled": True}
    return probe_path(PathBody(path=path))


@app.post("/api/video")
def probe_path(body: PathBody) -> Dict[str, Any]:
    path = body.path.strip().strip('"').strip("'")
    if not path:
        raise HTTPException(400, "No path given")
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"File not found: {path}")
    if not is_media_file(path):
        log.warning("Unusual extension for %s - trying anyway", path)
    try:
        info = ffmpeg().probe(str(p))
    except FFmpegError as exc:
        raise HTTPException(400, str(exc))
    cache = CacheStore(CFG.cache_dir)
    from ..cache.store import fingerprint
    vhash = fingerprint(str(p))
    cached = cache.load_transcript(vhash, str(CFG.get("whisper.model")),
                                   str(CFG.get("general.language"))) is not None
    return {"cancelled": False, "video": info.to_dict(),
            "cached_transcript": cached, "hash": vhash}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@app.post("/api/analyze")
def analyze(body: AnalyzeBody) -> Dict[str, Any]:
    if not Path(body.path).exists():
        raise HTTPException(404, "File not found")
    job = jobs.create(body.path)
    thread = threading.Thread(
        target=run_job, args=(CFG, job, jobs),
        kwargs={"force_transcribe": body.force_transcribe}, daemon=True)
    thread.start()
    return {"job_id": job.id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job.snapshot()


@app.post("/api/job/{job_id}/cancel")
def job_cancel(job_id: str) -> Dict[str, Any]:
    return {"ok": jobs.cancel(job_id)}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _result_of(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if not job or not job.result:
        raise HTTPException(404, "No finished analysis for that job")
    return job.result


@app.post("/api/export")
def export(body: ExportBody) -> Dict[str, Any]:
    result = _result_of(body.job_id)
    written = write_all(result, CFG.output_dir)
    return {"ok": True, "files": {k: v for k, v in written.items() if k in body.formats},
            "folder": str(CFG.output_dir)}


@app.get("/api/export/{job_id}.{fmt}")
def export_inline(job_id: str, fmt: str):
    result = _result_of(job_id)
    name = Path(result.get("video", {}).get("filename", "clips")).stem
    if fmt == "json":
        return Response(to_json(result), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
    if fmt == "csv":
        return Response("﻿" + to_csv(result), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
    if fmt == "txt":
        return PlainTextResponse(to_text(result),
                                 headers={"Content-Disposition": f'attachment; filename="{name}.txt"'})
    raise HTTPException(400, "Unsupported format")


@app.post("/api/export_clips")
def export_clips_endpoint(body: ExportClipsBody) -> Dict[str, Any]:
    result = _result_of(body.job_id)
    clips = result.get("clips", [])
    if body.ranks:
        wanted = set(body.ranks)
        clips = [c for c in clips if c.get("rank") in wanted]
    if not clips:
        raise HTTPException(400, "No clips selected")
    video = result.get("video", {})
    exp = CFG.section("export")
    caps = dict(CFG.section("captions"))
    if body.captions is not None:
        caps["enabled"] = bool(body.captions)
    fonts = CFG.fonts_dir
    try:
        written = export_clips(
            ffmpeg(), video.get("path"), clips, CFG.output_dir,
            mode=(body.mode or exp.get("mode", "copy")),
            crf=int(exp.get("reencode_crf", 20)),
            preset=str(exp.get("reencode_preset", "veryfast")),
            pad_start=float(exp.get("padding_start", 0.15)),
            pad_end=float(exp.get("padding_end", 0.35)),
            video_duration=float(video.get("duration", 0) or 0),
            captions=caps,
            video_width=int(video.get("width", 0) or 0),
            video_height=int(video.get("height", 0) or 0),
            encoder=str(exp.get("encoder", "auto")),
            fonts_dir=str(fonts) if fonts else None,
        )
    except FFmpegError as exc:
        raise HTTPException(500, str(exc))
    return {"ok": True, "clips": written,
            "folder": str(Path(written[0]["path"]).parent) if written else str(CFG.output_dir)}


@app.post("/api/caption_preview")
def caption_preview(body: CaptionPreviewBody):
    """Render a few seconds of one clip with captions burned in.

    Choosing a font and colours by re-exporting twelve clips each time would be
    miserable, so this renders one short sample instead.
    """
    job = jobs.get(body.job_id)
    if not job or not job.result:
        raise HTTPException(404, "No finished analysis for that job")
    clip = next((c for c in job.result.get("clips", []) if c.get("rank") == body.rank), None)
    if clip is None:
        raise HTTPException(404, "No such clip")

    video = job.result.get("video", {})
    exp = CFG.section("export")
    caps = {**CFG.section("captions"), "enabled": True}
    pad = float(exp.get("padding_start", 0.15))
    start = max(0.0, float(clip["start"]) - pad)
    end = min(float(clip["end"]), start + max(2.0, float(body.seconds)))

    tmp = Path(tempfile.mkdtemp(prefix="capprev_"))
    ass = write_ass_for_clip(_clip_words(clip), start, end,
                             int(video.get("width", 0) or 0),
                             int(video.get("height", 0) or 0),
                             caps, tmp / "p.ass")
    out = tmp / "preview.mp4"
    fonts = CFG.fonts_dir
    try:
        ffmpeg().burn(video.get("path"), str(out), start, end, str(ass),
                      encoder=str(exp.get("encoder", "auto")),
                      fonts_dir=str(fonts) if fonts else None)
    except FFmpegError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(500, str(exc))

    def stream():
        try:
            with open(out, "rb") as fh:
                while chunk := fh.read(262144):
                    yield chunk
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return StreamingResponse(stream(), media_type="video/mp4",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/fonts")
def fonts() -> Dict[str, Any]:
    """Caption font choices: known Windows faces plus anything dropped into
    assets/fonts/."""
    builtin = ["Arial Rounded MT Bold", "Segoe UI Black", "Arial Black", "Impact",
               "Bahnschrift", "Cooper Black", "Segoe UI Semibold", "Verdana"]
    dropped: List[str] = []
    raw = CFG.get("captions.fonts_dir", "")
    if raw:
        d = Path(raw) if Path(raw).is_absolute() else (ROOT_DIR / raw)
        if d.is_dir():
            dropped = sorted({p.stem for p in d.glob("*.[ot]t[fc]")})
    return {"builtin": builtin, "dropped": dropped,
            "folder": str((ROOT_DIR / raw) if raw else "")}


@app.post("/api/open_folder")
def open_folder(body: PathBody) -> Dict[str, Any]:
    target = Path(body.path or str(CFG.output_dir))
    if not target.exists():
        target = CFG.output_dir
    try:
        if os.name == "nt":
            os.startfile(str(target))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc))
    return {"ok": True}


# ---------------------------------------------------------------------------
# Media streaming (preview)
# ---------------------------------------------------------------------------
def _range_response(path: Path, request: Request) -> Response:
    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(str(path), media_type=media_type,
                            headers={"Accept-Ranges": "bytes"})

    try:
        units, _, rng = range_header.partition("=")
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(416, "Bad range header")
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    length = end - start + 1

    def stream():
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(262144, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(stream(), status_code=206, media_type=media_type, headers={
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    })


@app.get("/api/media")
def media(path: str, request: Request):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "File not found")
    return _range_response(p, request)


@app.get("/api/preview")
def preview(path: str, start: float = 0.0, end: float = 0.0, height: int = 720):
    """Transcoded fallback for containers/codecs the browser cannot decode."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File not found")
    if end <= start:
        end = start + 30.0
    stream = ffmpeg().preview_stream(str(p), start, end, height=height)
    return StreamingResponse(stream, media_type="video/mp4")


@app.get("/api/thumb")
def thumb(path: str, at: float = 0.0):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File not found")
    cache_dir = CFG.cache_dir / "thumbs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{abs(hash((str(p), round(at, 1))))}.jpg"
    if not out.exists():
        if not ffmpeg().thumbnail(str(p), at, str(out)):
            raise HTTPException(500, "Could not generate thumbnail")
    return FileResponse(str(out), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@app.get("/api/cache")
def cache_list() -> Dict[str, Any]:
    store = CacheStore(CFG.cache_dir)
    entries = store.entries()
    return {"entries": entries,
            "total_mb": round(sum(e["size_mb"] for e in entries), 1),
            "folder": str(CFG.cache_dir)}


@app.delete("/api/cache")
def cache_clear(hash: Optional[str] = None) -> Dict[str, Any]:
    removed = CacheStore(CFG.cache_dir).clear(hash)
    return {"ok": True, "removed": removed}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse({"detail": str(exc)}, status_code=500)
