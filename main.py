#!/usr/bin/env python
"""Local AI Clip Finder - entry point.

  python main.py                       start the UI (default)
  python main.py analyze video.mp4     headless run, prints + exports results
  python main.py check                 environment / model diagnostics
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app import APP_NAME, __version__          # noqa: E402
from app.config import load_config             # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    import os
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("uvicorn.access", "faster_whisper", "httpx", "httpcore",
                  "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
def cmd_serve(args) -> int:
    import uvicorn
    from app.ui import server as server_module

    cfg = load_config(args.config)
    server_module.CFG = cfg
    host = args.host or cfg.get("server.host", "127.0.0.1")
    port = args.port or int(cfg.get("server.port", 8420))
    url = f"http://{host}:{port}"

    print(f"\n  {APP_NAME} v{__version__}")
    print(f"  Running 100% locally at  {url}")
    print("  Press Ctrl+C to stop.\n")

    if cfg.get("server.open_browser", True) and not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(server_module.app, host=host, port=port, log_level="warning")
    return 0


def cmd_analyze(args) -> int:
    from app.export.exporters import write_all
    from app.models import fmt_ts
    from app.pipeline import Job, Pipeline

    cfg = load_config(args.config)
    if args.clips:
        cfg.update({"clips": {"target_count": args.clips}})
    if args.language:
        cfg.update({"general": {"language": args.language}})
    if args.whisper_model:
        cfg.update({"whisper": {"model": args.whisper_model}})
    if args.llm_model:
        cfg.update({"llm": {"model": args.llm_model}})

    path = str(Path(args.video).expanduser().resolve())
    job = Job(id="cli", video_path=path)
    pipe = Pipeline(cfg, job)

    stop = threading.Event()

    def ticker():
        last = ""
        while not stop.is_set():
            line = f"  [{job.progress * 100:5.1f}%] {job.stage_label}: {job.message}"
            if line != last:
                print(line.ljust(110)[:110], end="\r", flush=True)
                last = line
            time.sleep(0.4)

    threading.Thread(target=ticker, daemon=True).start()
    try:
        result = pipe.run(path, force_transcribe=args.force_transcribe)
    finally:
        stop.set()
        print(" " * 110, end="\r")

    print(f"\n=== {len(result.clips)} clips · {result.video.filename} ===\n")
    for c in result.clips:
        s = c.scores
        print(f"#{c.rank} — {c.overall:.0f}/100   [{c.clip_type}]")
        print(f"{fmt_ts(c.start)} → {fmt_ts(c.end)}   ({c.duration:.0f}s)")
        print(f"  Hook {s.get('hook', 0):.0f}/25   Payoff {s.get('payoff', 0):.0f}/25   "
              f"Emotion {s.get('emotion', 0):.0f}/15   Curiosity {s.get('curiosity', 0):.0f}/15   "
              f"Standalone {s.get('standalone', 0):.0f}/10   Editability {s.get('editability', 0):.0f}/10")
        print(f"  Title : {c.title}")
        print(f"  Why   : {c.reason}")
        if args.transcripts:
            print(f"  Text  : {' '.join(c.text.split())[:600]}")
        print()

    written = write_all(result.to_dict(), cfg.output_dir)
    for fmt, p in written.items():
        print(f"  {fmt.upper():4s} -> {p}")
    return 0


def cmd_check(args) -> int:
    from app.llm.base import get_backend
    from app.transcription.cuda_setup import cuda_available, ensure_cuda_dlls
    from app.video.ffmpeg_tools import FFmpeg

    cfg = load_config(args.config)
    print(f"\n{APP_NAME} v{__version__} — environment check\n" + "-" * 52)
    print(f"Python           : {sys.version.split()[0]}")

    ff = FFmpeg(cfg.get("general.ffmpeg_path", ""), cfg.get("general.ffprobe_path", ""))
    print(f"FFmpeg           : {'OK  ' + ff.ffmpeg if ff.available() else 'MISSING'}")

    added = ensure_cuda_dlls()
    print(f"CUDA (CTranslate2): {'available' if cuda_available() else 'not available (CPU mode)'}")
    if added:
        print(f"  CUDA DLL paths : {len(added)} folder(s) registered")

    print(f"Whisper model    : {cfg.get('whisper.model')} (device={cfg.get('whisper.device')})")

    try:
        h = get_backend(cfg.section("llm")).health()
        status = "OK" if h.get("ok") and h.get("model_installed") is not False else "PROBLEM"
        print(f"LLM backend      : {status} — {h.get('backend')} @ {h.get('url')}")
        print(f"LLM model        : {cfg.get('llm.model')}")
        if h.get("error"):
            print(f"  ! {h['error']}")
        if h.get("models"):
            print(f"  installed      : {', '.join(h['models'][:12])}")
    except Exception as exc:  # noqa: BLE001
        print(f"LLM backend      : ERROR — {exc}")

    print(f"Cache dir        : {cfg.cache_dir}")
    print(f"Output dir       : {cfg.output_dir}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clipfinder", description=APP_NAME)
    p.add_argument("--config", default=None, help="path to a config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("serve", help="start the web UI (default)")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("analyze", help="analyse a video without the UI")
    a.add_argument("video")
    a.add_argument("--clips", type=int, default=None, help="how many clips to return")
    a.add_argument("--language", default=None, help="auto | en | fr | ...")
    a.add_argument("--whisper-model", default=None)
    a.add_argument("--llm-model", default=None)
    a.add_argument("--force-transcribe", action="store_true", help="ignore cached transcript")
    a.add_argument("--transcripts", action="store_true", help="print clip transcripts")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("check", help="verify FFmpeg / CUDA / LLM setup")
    c.set_defaults(func=cmd_check)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    if not getattr(args, "command", None):
        # No subcommand given -> serve, keeping any global flags the user passed.
        args = parser.parse_args(sys.argv[1:] + ["serve"])
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("clipfinder").error("%s", exc)
        if args.verbose:
            raise
        print(f"\nERROR: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
