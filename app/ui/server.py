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
from pydantic import BaseModel, field_validator

from .. import APP_NAME, __version__
from ..analytics.store import PerformanceStore
from ..cache.store import CacheStore
from ..config import Config, load_config
from ..export.exporters import (_clip_words, export_clips, export_compilation,
                                normalize_music, normalize_watermark, to_csv, to_json,
                                to_text, write_all, write_ass_for_clip)
from ..export.metadata import write_metadata_for_clips
from ..export.thumbnail import pil_available
from ..llm.base import get_backend
from ..models import fmt_ts
from ..pipeline import JobManager, Pipeline, run_job
from ..ranking.ranker import arrange_for_retention
from ..transcription.cuda_setup import cuda_available
from ..video.ffmpeg_tools import FFmpeg, FFmpegError, is_media_file
from ..video.reframe import (FORMATS, build_filter, plan_reframe,
                             subject_tracking_available)
from .filedialog import pick_file

log = logging.getLogger("clipfinder.server")

STATIC_DIR = Path(__file__).parent / "static"
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI(title=APP_NAME, version=__version__, docs_url=None, redoc_url=None)
jobs = JobManager()
CFG: Config = load_config()
PERF = PerformanceStore(CFG.performance_db)


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


def _clean_formats(value: Any) -> Optional[List[str]]:
    """Coerce and filter a requested format list to known ids.

    Deliberately tolerant: a stray entry should not 422 the whole export, it
    should just be ignored.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    out: List[str] = []
    for item in value:
        key = str(item).strip()
        if key in FORMATS and key not in out:
            out.append(key)
    return out or None


class ExportClipsBody(BaseModel):
    job_id: str
    ranks: Optional[List[int]] = None
    mode: Optional[str] = None
    captions: Optional[bool] = None
    formats: Optional[List[Any]] = None
    layout: Optional[str] = None
    generate_metadata: Optional[bool] = None
    campaign_rules: Optional[str] = None
    watermark: Optional[bool] = None
    tighten: Optional[bool] = None
    thumbnail: Optional[bool] = None
    music: Optional[bool] = None

    @field_validator("formats", mode="before")
    @classmethod
    def _fmts(cls, v):  # noqa: N805
        return _clean_formats(v)


class ExportCompilationBody(BaseModel):
    job_id: str
    ranks: Optional[List[int]] = None
    max_clips: int = 6
    captions: Optional[bool] = None
    formats: Optional[List[Any]] = None
    layout: Optional[str] = None
    name: Optional[str] = None
    generate_metadata: Optional[bool] = None
    campaign_rules: Optional[str] = None
    watermark: Optional[bool] = None
    tighten: Optional[bool] = None
    music: Optional[bool] = None

    @field_validator("formats", mode="before")
    @classmethod
    def _fmts(cls, v):  # noqa: N805
        return _clean_formats(v)


class CaptionPreviewBody(BaseModel):
    job_id: str
    rank: int = 1
    seconds: float = 6.0
    format: Optional[str] = None
    layout: Optional[str] = None
    captions: bool = True
    watermark: Optional[bool] = None
    music: Optional[bool] = None


class ConfigBody(BaseModel):
    config: Dict[str, Any]
    save: bool = True


class PerformanceBody(BaseModel):
    job_id: Optional[str] = None
    rank: Optional[str] = None
    video_hash: Optional[str] = None
    video_name: Optional[str] = None
    title: Optional[str] = None
    clip_type: Optional[str] = None
    duration: Optional[float] = None
    scores: Optional[Dict[str, float]] = None
    overall: Optional[float] = None
    virality: Optional[float] = None
    platform: str = "other"
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    rating: Optional[float] = None
    notes: Optional[str] = None
    posted_at: Optional[str] = None


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
        "pillow": {"ok": pil_available()},
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


@app.post("/api/browse_image")
def browse_image() -> Dict[str, Any]:
    try:
        path = pick_file(kind="image")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Could not open the file picker: {exc}")
    if not path:
        return {"cancelled": True}
    if not Path(path).is_file():
        raise HTTPException(404, f"File not found: {path}")
    return {"cancelled": False, "path": path}


@app.post("/api/browse_audio")
def browse_audio() -> Dict[str, Any]:
    try:
        path = pick_file(kind="audio")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Could not open the file picker: {exc}")
    if not path:
        return {"cancelled": True}
    if not Path(path).is_file():
        raise HTTPException(404, f"File not found: {path}")
    return {"cancelled": False, "path": path}


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
    running = jobs.active()
    if running is not None:
        raise HTTPException(409, (
            f"An analysis is already running ({Path(running.video_path).name}, "
            f"{running.stage_label}). Whisper and the LLM each want the whole GPU, so "
            "running two at once is slower than waiting. Cancel it first, or let it finish."
        ))
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


def _watermark_cfg(override: Optional[bool]) -> Dict[str, Any]:
    """The saved watermark settings, with just `enabled` overridable per export
    request (the image/position/size are configured once, not per click)."""
    wm = dict(CFG.section("watermark"))
    if override is not None:
        wm["enabled"] = bool(override)
    return wm


def _tighten_cfg(override: Optional[bool]) -> Dict[str, Any]:
    tg = dict(CFG.section("tightening"))
    if override is not None:
        tg["enabled"] = bool(override)
    return tg


def _thumbnail_cfg(override: Optional[bool]) -> Dict[str, Any]:
    th = dict(CFG.section("thumbnail"))
    if override is not None:
        th["enabled"] = bool(override)
    return th


def _music_cfg(override: Optional[bool]) -> Dict[str, Any]:
    mu = dict(CFG.section("music"))
    if override is not None:
        mu["enabled"] = bool(override)
    return mu


def _maybe_write_metadata(result: Dict[str, Any], clips: List[Dict[str, Any]],
                          folder: Path, want: Optional[bool],
                          campaign_rules: Optional[str]) -> Optional[Dict[str, Any]]:
    """Write a `<clip>_metadata.txt` per clip. Never raises - a metadata miss
    is a lesser outcome than losing the clips themselves - but always reports
    back what happened (or why nothing was written) so the caller can tell
    the user, instead of a silent no-op.
    """
    camp = CFG.section("campaign")
    if want is None:
        want = bool(camp.get("generate_metadata", False))
    if not want:
        return None
    rules = campaign_rules if campaign_rules is not None else str(camp.get("rules", ""))
    try:
        backend = get_backend(CFG.section("llm"))
        health = backend.health()
        if not health.get("ok"):
            msg = f"Metadata skipped: local LLM unreachable ({health.get('error') or 'offline'})"
            log.warning(msg)
            return {"requested": len(clips), "written": [], "failed": [], "error": msg}
        if health.get("model_installed") is False:
            msg = f"Metadata skipped: {health.get('error') or 'model not installed'}"
            log.warning(msg)
            return {"requested": len(clips), "written": [], "failed": [], "error": msg}
    except Exception as exc:  # noqa: BLE001
        msg = f"Metadata skipped: {exc}"
        log.warning(msg)
        return {"requested": len(clips), "written": [], "failed": [], "error": msg}

    language = str(result.get("language") or CFG.get("general.language") or "en")
    try:
        outcome = write_metadata_for_clips(backend, clips, folder, language=language,
                                           campaign_rules=rules, on_log=log.info)
        log.info("Wrote metadata for %d/%d clip(s)", len(outcome["written"]), len(clips))
        return {"requested": len(clips), "written": outcome["written"],
                "failed": outcome["failed"], "error": None}
    except Exception as exc:  # noqa: BLE001
        msg = f"Metadata generation failed: {exc}"
        log.warning(msg)
        return {"requested": len(clips), "written": [], "failed": [], "error": msg}


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
    rf = CFG.section("reframe")
    formats = body.formats or rf.get("formats") or ["9:16"]
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
            formats=formats,
            layout=str(body.layout or rf.get("layout", "crop")),
            sample_fps=float(rf.get("sample_fps", 3.0)),
            watermark=_watermark_cfg(body.watermark),
            tighten=_tighten_cfg(body.tighten),
            thumbnail=_thumbnail_cfg(body.thumbnail),
            music=_music_cfg(body.music),
            on_log=log.info,
        )
    except FFmpegError as exc:
        raise HTTPException(500, str(exc))
    folder = Path(written[0]["path"]).parent if written else CFG.output_dir
    metadata = _maybe_write_metadata(result, clips, folder, body.generate_metadata,
                                     body.campaign_rules)
    return {"ok": True, "clips": written, "folder": str(folder), "metadata": metadata}


@app.post("/api/export_compilation")
def export_compilation_endpoint(body: ExportCompilationBody) -> Dict[str, Any]:
    """Stitch several clips into ONE viral-format video, auto-arranged to open
    on the strongest hook, close on the biggest payoff, and keep the middle
    from sagging in between."""
    result = _result_of(body.job_id)
    clips = result.get("clips", [])
    if body.ranks:
        wanted = set(body.ranks)
        clips = [c for c in clips if c.get("rank") in wanted]
    else:
        # No explicit picks - take the highest-virality moments so this works
        # as a one-click action, not just for a pre-selected set.
        clips = sorted(clips, key=lambda c: -(c.get("virality") or c.get("score", 0)))
        clips = clips[: max(2, int(body.max_clips))]
    if len(clips) < 2:
        raise HTTPException(400, "Pick at least 2 clips to build a compilation")

    ordered = arrange_for_retention(clips)

    video = result.get("video", {})
    exp = CFG.section("export")
    caps = dict(CFG.section("captions"))
    if body.captions is not None:
        caps["enabled"] = bool(body.captions)
    rf = CFG.section("reframe")
    formats = body.formats or rf.get("formats") or ["9:16"]
    name = body.name or f"{Path(video.get('filename', 'clips')).stem}_viral_cut"
    try:
        written = export_compilation(
            ffmpeg(), video.get("path"), ordered, CFG.output_dir,
            name=name,
            pad_start=float(exp.get("padding_start", 0.15)),
            pad_end=float(exp.get("padding_end", 0.35)),
            video_duration=float(video.get("duration", 0) or 0),
            captions=caps,
            video_width=int(video.get("width", 0) or 0),
            video_height=int(video.get("height", 0) or 0),
            encoder=str(exp.get("encoder", "auto")),
            formats=formats,
            layout=str(body.layout or rf.get("layout", "crop")),
            sample_fps=float(rf.get("sample_fps", 3.0)),
            watermark=_watermark_cfg(body.watermark),
            tighten=_tighten_cfg(body.tighten),
            music=_music_cfg(body.music),
            on_log=log.info,
        )
    except FFmpegError as exc:
        raise HTTPException(500, str(exc))
    folder = Path(written[0]["path"]).parent if written else CFG.output_dir
    # The compilation is ONE video stitched from several clips, so it gets ONE
    # metadata doc describing the whole thing rather than one per source clip.
    combo = {
        "rank": "compilation", "title": name,
        "start_tc": "", "end_tc": "",
        "duration": sum(float(c.get("duration", 0) or 0) for c in ordered),
        "type": "compilation",
        "transcript": " ".join(str(c.get("transcript", "")) for c in ordered),
    }
    metadata = _maybe_write_metadata(result, [combo], folder, body.generate_metadata,
                                     body.campaign_rules)
    return {"ok": True, "files": written, "folder": str(folder), "metadata": metadata}


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
    rf = CFG.section("reframe")
    caps = {**CFG.section("captions"), "enabled": bool(body.captions)}
    pad = float(exp.get("padding_start", 0.15))
    start = max(0.0, float(clip["start"]) - pad)
    end = min(float(clip["end"]), start + max(2.0, float(body.seconds)))

    src_w = int(video.get("width", 0) or 0)
    src_h = int(video.get("height", 0) or 0)
    fmt = body.format or (rf.get("formats") or ["9:16"])[0]
    layout = str(body.layout or rf.get("layout", "crop"))
    watermark = normalize_watermark(_watermark_cfg(body.watermark))
    music = normalize_music(_music_cfg(body.music))

    tmp = Path(tempfile.mkdtemp(prefix="capprev_"))
    out = tmp / "preview.mp4"
    try:
        path = plan_reframe(ffmpeg().ffmpeg, video.get("path"), start, end, fmt,
                            src_w, src_h, layout=layout,
                            sample_fps=float(rf.get("sample_fps", 3.0)),
                            on_log=log.info)
        sub = None
        if caps["enabled"]:
            _, ow, oh, _, _, _ = build_filter(src_w, src_h, fmt, path, layout=layout)
            write_ass_for_clip(_clip_words(clip), start, end, ow, oh, caps, tmp / "p.ass")
            sub = "p.ass"
        vf, _, _, use_fc, extra_inputs, audio_map = build_filter(
            src_w, src_h, fmt, path, layout=layout, subtitle_file=sub,
            watermark=watermark, music=music, clip_duration=end - start)
        ffmpeg().render(video.get("path"), str(out), start, end, vf=vf,
                        work_dir=str(tmp), encoder=str(exp.get("encoder", "auto")),
                        extra_inputs=extra_inputs, filter_complex=use_fc,
                        audio_map=audio_map)
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


@app.get("/api/formats")
def formats() -> Dict[str, Any]:
    """Aspect-ratio presets for the export picker."""
    rf = CFG.section("reframe")
    return {
        "formats": [
            {"id": key, "width": w, "height": h, "label": label, "platforms": plat}
            for key, (w, h, label, plat) in FORMATS.items()
        ],
        "selected": rf.get("formats") or ["9:16"],
        "layout": rf.get("layout", "crop"),
        "tracking": subject_tracking_available(),
    }


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
# Performance tracking - log how exported clips actually did, and surface
# which score categories correlate with real engagement.
# ---------------------------------------------------------------------------
def _clip_snapshot(job_id: Optional[str], rank: Optional[str]) -> Dict[str, Any]:
    """Best-effort clip metadata (title/type/scores/...) so a logged entry is
    still self-describing even after the job is pruned from memory."""
    if not job_id or rank is None:
        return {}
    job = jobs.get(job_id)
    if not job or not job.result:
        return {}
    clip = next((c for c in job.result.get("clips", []) if str(c.get("rank")) == str(rank)), None)
    if not clip:
        return {}
    video = job.result.get("video", {})
    return {
        "video_hash": job.result.get("stats", {}).get("video_hash"),
        "video_name": video.get("filename"),
        "title": clip.get("title"),
        "clip_type": clip.get("type"),
        "duration": clip.get("duration"),
        "scores": clip.get("scores"),
        "overall": clip.get("score"),
        "virality": clip.get("virality"),
    }


@app.post("/api/performance")
def log_performance(body: PerformanceBody) -> Dict[str, Any]:
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    snapshot = _clip_snapshot(body.job_id, body.rank)
    entry = {**snapshot, **data}   # explicit request fields win over the snapshot
    entry_id = PERF.log(entry)
    return {"ok": True, "id": entry_id}


@app.get("/api/performance")
def list_performance(video_hash: Optional[str] = None) -> Dict[str, Any]:
    return {"entries": PERF.list(video_hash=video_hash)}


@app.delete("/api/performance/{entry_id}")
def delete_performance(entry_id: int) -> Dict[str, Any]:
    return {"ok": PERF.delete(entry_id)}


@app.get("/api/performance/insights")
def performance_insights() -> Dict[str, Any]:
    return PERF.insights(current_weights=CFG.weights)


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
