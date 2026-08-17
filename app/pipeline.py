"""The orchestrator: video in, ranked clip candidates out."""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .cache.store import CacheStore, fingerprint
from .candidates.generator import generate_windows, hotspots_in_block
from .config import Config
from .llm.base import LLMBackend, get_backend
from .models import AnalysisResult, Candidate, Sentence, Transcript, VideoInfo
from .ranking.ranker import dedupe, second_pass, select_diverse, type_breakdown
from .scoring.scorer import (analyze_block, apply_padding, heuristic_only_scores,
                             optimize_boundaries, trim_edges)
from .segmentation.segmenter import build_sentences, split_into_blocks, topic_shift_scores
from .transcription.whisper_engine import WhisperEngine
from .video.ffmpeg_tools import FFmpeg, FFmpegError

log = logging.getLogger("clipfinder.pipeline")

# Fraction of the overall progress bar each stage owns.
STAGE_WEIGHTS = {
    "probe": 0.01,
    "audio": 0.04,
    "transcribe": 0.45,
    "segment": 0.03,
    "pass1": 0.32,
    "boundaries": 0.09,
    "pass2": 0.05,
    "finalize": 0.01,
}
STAGE_ORDER = list(STAGE_WEIGHTS)


class Cancelled(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    video_path: str
    status: str = "queued"          # queued | running | done | error | cancelled
    stage: str = "queued"
    stage_label: str = "Queued"
    progress: float = 0.0           # 0..1 overall
    message: str = ""
    logs: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    stage_started: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    video: Optional[Dict[str, Any]] = None

    def snapshot(self) -> Dict[str, Any]:
        now = time.time() if self.status == "running" else (self.finished_at or time.time())
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "progress": round(self.progress, 4),
            "message": self.message,
            # Lets the UI prove the run is alive during a long single step
            # (a cold model load, one big transcript block) instead of looking frozen.
            "stage_elapsed": round(now - self.stage_started, 1),
            "since_update": round(now - self.last_update, 1),
            "logs": self.logs[-200:],
            "error": self.error,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 1),
            "video": self.video,
            # The result is large; only ship it once, on the final poll.
            "result": self.result if self.status == "done" else None,
        }


STAGE_LABELS = {
    "probe": "Reading video",
    "audio": "Extracting audio",
    "transcribe": "Transcribing (Whisper)",
    "segment": "Segmenting transcript",
    "pass1": "Pass 1 · finding moments",
    "boundaries": "Optimising clip boundaries",
    "pass2": "Pass 2 · ranking head-to-head",
    "finalize": "Finalising",
}


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, video_path: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], video_path=video_path)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job and job.status in ("queued", "running"):
            job.cancel_event.set()
            job.message = "Cancelling…"
            return True
        return False

    def prune(self, keep: int = 12) -> None:
        with self._lock:
            if len(self._jobs) <= keep:
                return
            done = sorted((j for j in self._jobs.values()
                           if j.status in ("done", "error", "cancelled")),
                          key=lambda j: j.finished_at)
            for j in done[: max(0, len(self._jobs) - keep)]:
                self._jobs.pop(j.id, None)


class Pipeline:
    def __init__(self, cfg: Config, job: Optional[Job] = None):
        self.cfg = cfg
        self.job = job
        self.ff = FFmpeg(cfg.get("general.ffmpeg_path", ""), cfg.get("general.ffprobe_path", ""))
        self.cache = CacheStore(cfg.cache_dir)
        self._stage_base = 0.0

    # -------------------------------------------------------------- progress
    def _check_cancel(self) -> None:
        if self.job and self.job.cancel_event.is_set():
            raise Cancelled()

    def log(self, message: str) -> None:
        log.info(message)
        if self.job:
            stamp = time.strftime("%H:%M:%S")
            self.job.logs.append(f"[{stamp}] {message}")
            self.job.last_update = time.time()

    def stage(self, name: str, message: str = "") -> None:
        self._stage_base = sum(STAGE_WEIGHTS[s] for s in STAGE_ORDER[:STAGE_ORDER.index(name)])
        if self.job:
            self.job.stage = name
            self.job.stage_label = STAGE_LABELS.get(name, name)
            self.job.progress = self._stage_base
            self.job.message = message or STAGE_LABELS.get(name, "")
            self.job.stage_started = time.time()
            self.job.last_update = time.time()
        if message:
            self.log(message)

    def tick(self, fraction: float, message: str = "") -> None:
        self._check_cancel()
        if not self.job:
            return
        name = self.job.stage
        weight = STAGE_WEIGHTS.get(name, 0.0)
        self.job.progress = min(0.999, self._stage_base + weight * max(0.0, min(1.0, fraction)))
        self.job.last_update = time.time()
        if message:
            self.job.message = message

    # ------------------------------------------------------- speech sanity
    def _require_speech(self, sentences: List[Sentence], info: VideoInfo) -> None:
        """Fail with a diagnosis, not a shrug, when there is nothing to analyse."""
        words = sum(s.word_count for s in sentences)
        spoken = sum(s.duration for s in sentences)
        ratio = (spoken / info.duration) if info.duration > 0 else 0.0
        self.log(f"Speech found: {words} words across {len(sentences)} units "
                 f"({ratio * 100:.0f}% of the video)")

        if len(sentences) >= 4 and words >= 25:
            return

        if words == 0:
            detail = ("Whisper found no speech at all. The audio track is present but "
                      "contains no recognisable dialogue.")
        else:
            detail = (f"Whisper found only {words} word(s) in "
                      f"{info.duration / 60:.1f} minutes, which is noise, not dialogue.")

        raise RuntimeError(
            f"{detail}\n\n"
            "This tool finds clips from what people SAY, so it needs a video with "
            "talking in it - a podcast, interview, stream commentary, lecture or "
            "vlog. Gameplay footage, music videos and b-roll have nothing for it "
            "to read.\n\n"
            "If you are sure there IS speech in this file:\n"
            "  - check the audio actually plays (some screen recordings capture a "
            "silent or wrong audio device)\n"
            "  - if the speech sits under loud music, set whisper.vad_filter: false "
            "in config.yaml\n"
            "  - try a bigger model, whisper.model: turbo, for quiet or noisy audio"
        )

    # ------------------------------------------------------------------- run
    def run(self, video_path: str, *, force_transcribe: bool = False) -> AnalysisResult:
        clips_cfg = self.cfg.section("clips")
        weights = self.cfg.weights
        blend = float(self.cfg.get("scoring.heuristic_blend", 0.15))
        language_cfg = str(self.cfg.get("general.language", "auto"))

        # ---- 1. probe -----------------------------------------------------
        self.stage("probe", f"Reading {Path(video_path).name}")
        if not self.ff.available():
            raise FFmpegError(
                "FFmpeg is not installed or not on PATH. Run install.bat, or install it "
                "manually with: winget install Gyan.FFmpeg"
            )
        info = self.ff.probe(video_path)
        if self.job:
            self.job.video = info.to_dict()
        if not info.has_audio:
            raise FFmpegError("This file has no audio track - there is nothing to transcribe.")

        min_d = float(clips_cfg.get("min_duration", 20))
        if info.duration < min_d:
            raise RuntimeError(
                f"This video is only {info.duration:.0f}s long, shorter than the "
                f"{min_d:.0f}s minimum clip length, so no clip can be cut from it. "
                "Lower clips.min_duration, or analyse the longer video this was cut from."
            )
        if info.duration < min_d * 3:
            self.log(f"Heads up: {info.duration:.0f}s of video can hold at most one "
                     f"{min_d:.0f}-{clips_cfg.get('max_duration', 90)}s clip. This tool is "
                     "built for long videos you want to mine for short moments.")
        self.log(f"{info.filename} · {info.duration / 60:.1f} min · "
                 f"{info.width}x{info.height} · {info.fps:.0f} fps")

        vhash = fingerprint(video_path)
        self.cache.write_metadata(vhash, info)

        # ---- 2 & 3. audio + transcription (cached) ------------------------
        whisper_cfg = self.cfg.section("whisper")
        model_name = str(whisper_cfg.get("model", "small"))
        transcript: Optional[Transcript] = None
        if not force_transcribe:
            transcript = self.cache.load_transcript(vhash, model_name, language_cfg)
            if transcript:
                self.stage("transcribe", "Loaded transcript from cache (no re-transcription)")
                self.tick(1.0)

        if transcript is None:
            self.stage("audio", "Extracting 16 kHz mono audio")
            audio_path = self.ff.extract_audio(video_path)
            self.tick(1.0)
            engine = WhisperEngine(whisper_cfg, cache_dir=str(self.cfg.cache_dir / "models"))
            try:
                self.stage("transcribe")
                self._check_cancel()
                transcript = engine.transcribe(
                    audio_path,
                    language=language_cfg,
                    duration_hint=info.duration,
                    on_progress=lambda f, m: self.tick(f, m),
                    on_log=self.log,
                )
                self.log(f"Transcribed on {engine.device.upper()} · "
                         f"{len(transcript.raw_segments)} segments")
            finally:
                engine.unload()   # free VRAM before the LLM stage
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            self.cache.save_transcript(vhash, model_name, language_cfg, transcript)

        language = transcript.language or "en"

        # ---- 4. segmentation + offline signals ----------------------------
        self.stage("segment", "Building natural speech units")
        sentences = build_sentences(transcript)
        transcript.sentences = sentences
        self._require_speech(sentences, info)
        self.tick(0.35)

        topic = topic_shift_scores(sentences)
        heur_cands, profiles, window_scorer = generate_windows(
            sentences, language,
            min_duration=float(clips_cfg.get("min_duration", 20)),
            max_duration=float(clips_cfg.get("max_duration", 90)),
            ideal_duration=float(clips_cfg.get("ideal_duration", 45)),
            topic=topic,
        )
        self.log(f"{len(sentences)} speech units · {len(heur_cands)} heuristic windows")
        self.tick(1.0)

        blocks = split_into_blocks(
            sentences,
            float(clips_cfg.get("block_seconds", 420)),
            float(clips_cfg.get("block_overlap_seconds", 45)),
        )
        max_blocks = int(clips_cfg.get("max_blocks", 0) or 0)
        if max_blocks > 0:
            blocks = blocks[:max_blocks]

        # ---- 5. pass 1 ----------------------------------------------------
        # Everything LLM-related lives inside the pass1 stage, including the
        # connection check and the model load - those take real time and must
        # not be reported under the previous stage's label.
        self.stage("pass1", "Connecting to the local LLM")
        backend: Optional[LLMBackend] = None
        llm_ok = False
        try:
            backend = get_backend(self.cfg.section("llm"))
            health = backend.health()
            llm_ok = bool(health.get("ok"))
            if not llm_ok:
                self.log(f"LLM unavailable: {health.get('error')}")
            elif health.get("model_installed") is False:
                self.log(health.get("error") or "Model missing")
                llm_ok = False
            else:
                self.log(f"LLM ready: {health.get('model')} via {health.get('backend')}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"LLM backend error: {exc}")

        raw_candidates: List[Candidate] = []

        if llm_ok and backend is not None:
            model_name = str(self.cfg.get("llm.model", "model"))
            self.tick(0.0, f"Loading {model_name} into VRAM (first call is the slow one)")
            warm_started = time.time()
            if backend.warmup():
                self.log(f"Model loaded in {time.time() - warm_started:.0f}s")

            total_blocks = max(1, len(blocks))
            for n, block in enumerate(blocks, start=1):
                self._check_cancel()
                t0 = sentences[block[0]].start
                where = f"{int(t0 // 60):02d}:{int(t0 % 60):02d}"
                # Tick BEFORE the call: a block can take a minute and the user
                # needs to see which one is being worked on, not which one ended.
                self.tick((n - 1) / total_blocks,
                          f"Block {n}/{len(blocks)} · from {where} · reading…")
                hints = hotspots_in_block(heur_cands, block[0], block[1])
                found = analyze_block(
                    backend, sentences, block, language, clips_cfg, weights,
                    hints=hints, window_scorer=window_scorer,
                    heuristic_blend=blend, on_log=self.log,
                )
                raw_candidates.extend(found)
                self.tick(n / total_blocks,
                          f"Block {n}/{len(blocks)} · from {where}"
                          f" · {len(raw_candidates)} candidates so far")
            self.log(f"Pass 1 found {len(raw_candidates)} candidates")

        if not raw_candidates:
            reason = "no local LLM" if not llm_ok else "the LLM returned nothing usable"
            self.log(f"Falling back to the offline detector ({reason}).")
            for cand in heur_cands[: int(clips_cfg.get("max_candidates_pass1", 60))]:
                raw_candidates.append(heuristic_only_scores(cand, window_scorer, weights))
        self.tick(1.0)

        cap = int(clips_cfg.get("max_candidates_pass1", 60))
        raw_candidates.sort(key=lambda c: -c.overall)
        raw_candidates = raw_candidates[:cap]

        # ---- 6. boundary optimisation -------------------------------------
        self.stage("boundaries")
        max_d = float(clips_cfg.get("max_duration", 90))
        pre = dedupe(raw_candidates, sentences,
                     float(clips_cfg.get("overlap_iou_threshold", 0.4)), max_d)
        self.log(f"{len(pre)} candidates after overlap merging")

        if llm_ok and backend is not None and clips_cfg.get("boundary_optimization", True):
            todo = pre[: int(clips_cfg.get("boundary_candidates", 20))]
            for n, cand in enumerate(todo, start=1):
                self._check_cancel()
                optimize_boundaries(backend, sentences, cand, clips_cfg,
                                    window_scorer=window_scorer, language=language,
                                    on_log=self.log)
                self.tick(n / max(1, len(todo)), f"Tuning boundaries {n}/{len(todo)}")
            moved = sum(1 for c in todo if c.boundary_optimized)
            self.log(f"Boundary pass reviewed {moved} clips")
            # merging can happen again once boundaries moved
            pre = dedupe(pre, sentences,
                         float(clips_cfg.get("overlap_iou_threshold", 0.4)), max_d)
        self.tick(1.0)

        # ---- 7. pass 2 ----------------------------------------------------
        self.stage("pass2")
        target = int(clips_cfg.get("target_count", 12))
        if llm_ok and backend is not None and len(pre) > 1:
            self.tick(0.2, "Comparing the finalists")
            pre = second_pass(backend, pre, language, target,
                              pool_size=int(clips_cfg.get("final_pool_size", 18)),
                              on_log=self.log)
            self.log(f"Pass 2 kept {len(pre)} clips")
        self.tick(1.0)

        # ---- 8. diversity + finalise --------------------------------------
        self.stage("finalize", "Selecting a varied final set")
        final = select_diverse(
            pre, target, info.duration,
            lam=float(clips_cfg.get("diversity_lambda", 0.7)),
            min_gap=float(clips_cfg.get("min_gap_between_clips", 5)),
        )
        trimmed = 0
        for cand in final:
            if trim_edges(cand, sentences, float(clips_cfg.get("min_duration", 20))):
                trimmed += 1
            apply_padding(cand, sentences)
            if not cand.title:
                cand.title = (cand.text.split(".")[0] or cand.text)[:70].strip()
            if not cand.hook_line:
                cand.hook_line = sentences[cand.start_idx].text[:200]
        if trimmed:
            self.log(f"Trimmed filler/transition lines off {trimmed} clip(s)")

        if len(final) < target:
            self.log(
                f"Returned {len(final)} of the {target} clips requested - the rest of "
                "the video did not contain moments that stand on their own. Raise "
                "clips.max_candidates_pass1 or lower clips.min_gap_between_clips to widen the net."
            )

        if backend is not None and self.cfg.get("llm.unload_after_analysis", True):
            backend.unload()

        stats = {
            "requested": target,
            "sentences": len(sentences),
            "blocks": len(blocks),
            "candidates_pass1": len(raw_candidates),
            "candidates_after_merge": len(pre),
            "final": len(final),
            "llm_used": llm_ok,
            "llm_model": self.cfg.get("llm.model") if llm_ok else None,
            "whisper_model": model_name,
            "types": type_breakdown(final),
            "elapsed": round(time.time() - (self.job.started_at if self.job else time.time()), 1),
            "video_hash": vhash,
        }
        result = AnalysisResult(video=info, clips=final, language=language, stats=stats)
        self.cache.save_result(vhash, result.to_dict())
        self.log(f"Done · {len(final)} clips in {stats['elapsed']:.0f}s")
        if self.job:
            self.job.progress = 1.0
        return result


def run_job(cfg: Config, job: Job, manager: JobManager, *,
            force_transcribe: bool = False) -> None:
    """Thread body. Never raises - failures land on the job object."""
    job.status = "running"
    pipe = Pipeline(cfg, job)
    try:
        result = pipe.run(job.video_path, force_transcribe=force_transcribe)
        job.result = result.to_dict()
        job.status = "done"
        job.stage_label = "Complete"
        job.message = f"{len(result.clips)} clips found"
        job.progress = 1.0
    except Cancelled:
        job.status = "cancelled"
        job.stage_label = "Cancelled"
        job.message = "Analysis cancelled"
        pipe.log("Analysis cancelled by user.")
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.stage_label = "Failed"
        job.error = str(exc)
        job.message = str(exc)
        pipe.log(f"ERROR: {exc}")
        log.exception("Analysis failed")
    finally:
        job.finished_at = time.time()
        manager.prune()
