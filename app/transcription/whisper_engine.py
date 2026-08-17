"""faster-whisper transcription with word-level timestamps and VRAM hygiene."""
from __future__ import annotations

import gc
import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..models import Transcript, Word
from .cuda_setup import cuda_available, ensure_cuda_dlls

log = logging.getLogger("clipfinder.whisper")

# Friendly aliases -> actual model ids understood by faster-whisper
MODEL_ALIASES = {
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "distil-large": "distil-large-v3",
}


class WhisperEngine:
    def __init__(self, cfg: Dict[str, Any], cache_dir: Optional[str] = None):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self._model = None
        self._device = "cpu"
        self._compute = "int8"

    # ------------------------------------------------------------- lifecycle
    def _resolve_device(self) -> tuple[str, str]:
        want = str(self.cfg.get("device", "auto")).lower()
        compute = str(self.cfg.get("compute_type", "auto")).lower()

        if want in ("auto", "cuda"):
            if cuda_available():
                device = "cuda"
            else:
                if want == "cuda":
                    log.warning("CUDA requested but not available to CTranslate2 - using CPU.")
                device = "cpu"
        else:
            device = "cpu"

        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return device, compute

    def load(self, on_log: Optional[Callable[[str], None]] = None):
        if self._model is not None:
            return self._model
        ensure_cuda_dlls()
        from faster_whisper import WhisperModel

        name = str(self.cfg.get("model", "small"))
        name = MODEL_ALIASES.get(name, name)
        device, compute = self._resolve_device()

        def emit(msg: str):
            log.info(msg)
            if on_log:
                on_log(msg)

        emit(f"Loading Whisper '{name}' on {device.upper()} ({compute})")
        kwargs: Dict[str, Any] = {"device": device, "compute_type": compute}
        if self.cache_dir:
            kwargs["download_root"] = self.cache_dir
        threads = int(self.cfg.get("cpu_threads", 0) or 0)
        if threads > 0:
            kwargs["cpu_threads"] = threads

        try:
            self._model = WhisperModel(name, **kwargs)
        except Exception as exc:
            if device == "cuda":
                emit(f"CUDA load failed ({type(exc).__name__}) - falling back to CPU/int8.")
                device, compute = "cpu", "int8"
                kwargs.update({"device": device, "compute_type": compute})
                self._model = WhisperModel(name, **kwargs)
            else:
                raise
        self._device, self._compute = device, compute
        return self._model

    def unload(self) -> None:
        """Free VRAM before the LLM stage starts - important on 8 GB cards."""
        if self._model is not None:
            try:
                del self._model
            finally:
                self._model = None
            gc.collect()
            try:  # only if torch happens to be installed alongside
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            log.info("Whisper unloaded, VRAM released.")

    @property
    def device(self) -> str:
        return self._device

    # ---------------------------------------------------------- transcription
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        duration_hint: float = 0.0,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Transcript:
        """Transcribe, falling back to CPU if CUDA blows up mid-stream.

        CTranslate2 produces segments lazily, so a missing cuBLAS/cuDNN DLL only
        surfaces while iterating - not when the model is constructed. Catching it
        here is what makes the CPU fallback actually work.
        """
        try:
            result = self._transcribe(audio_path, language, duration_hint,
                                      on_progress, on_log)
        except Exception as exc:  # noqa: BLE001
            if self._device != "cuda":
                raise
            msg = f"GPU transcription failed ({exc}). Retrying on the CPU…"
            log.warning(msg)
            if on_log:
                on_log(msg)
            self.unload()
            self.cfg = {**self.cfg, "device": "cpu", "compute_type": "int8"}
            result = self._transcribe(audio_path, language, duration_hint,
                                      on_progress, on_log)

        if self.cfg.get("fill_gaps", True) and result.raw_segments:
            self._fill_gaps(audio_path, result, on_log=on_log)

        # Silero VAD occasionally rejects an entire file - heavily compressed
        # audio, loud background music, unusual mic processing. If it left us
        # with nothing, try again with VAD off before declaring defeat.
        if not result.raw_segments and self.cfg.get("vad_filter", True):
            msg = "Voice detection found no speech. Retrying with VAD disabled…"
            log.warning(msg)
            if on_log:
                on_log(msg)
            self.cfg = {**self.cfg, "vad_filter": False}
            result = self._transcribe(audio_path, language, duration_hint,
                                      on_progress, on_log)
        return result

    def _fill_gaps(self, audio_path: str, transcript: Transcript,
                   on_log: Optional[Callable[[str], None]] = None) -> None:
        """Re-listen to stretches Whisper returned nothing for.

        Whisper sometimes skips a passage outright - measured on real footage:
        large-v3-turbo dropped 6.6 seconds of clear dialogue that `small`
        transcribed fine. The clip then plays with no captions over someone
        talking. So any silent-in-the-transcript stretch that is NOT silent in
        the audio gets transcribed again on its own, where there is no
        surrounding context for the model to run away from.
        """
        min_gap = float(self.cfg.get("gap_min_seconds", 2.5))
        noise_floor = float(self.cfg.get("gap_noise_db", -38.0))
        gaps = find_speech_gaps(transcript.raw_segments, transcript.duration, min_gap)
        if not gaps:
            return
        try:
            samples, rate = _read_wav_mono16(audio_path)
        except Exception:  # noqa: BLE001
            return
        if samples is None or not rate:
            return

        loud = [(a, b) for a, b in gaps if _rms_db(samples, rate, a, b) > noise_floor]
        if not loud:
            return
        if on_log:
            total = sum(b - a for a, b in loud)
            on_log(f"Re-listening to {len(loud)} silent stretch(es) "
                   f"({total:.0f}s) that still have audio in them")

        model = self.load(on_log=on_log)
        recovered: List[Dict[str, Any]] = []
        tmp_dir = Path(tempfile.mkdtemp(prefix="gapfill_"))
        try:
            for n, (a, b) in enumerate(loud[:40]):     # bounded: this costs time
                piece = tmp_dir / f"g{n}.wav"
                pad = 0.20                              # a hair of lead-in helps
                if not _slice_wav(audio_path, str(piece), max(0.0, a - pad), b + pad):
                    continue
                try:
                    segs, _ = model.transcribe(
                        str(piece),
                        language=transcript.language,
                        beam_size=int(self.cfg.get("beam_size", 5)),
                        word_timestamps=bool(self.cfg.get("word_timestamps", True)),
                        condition_on_previous_text=False,
                        vad_filter=False,     # the window is already known to be loud
                    )
                    offset = max(0.0, a - pad)
                    for seg in segs:
                        if _is_hallucination(seg):
                            continue
                        words = []
                        for w in (getattr(seg, "words", None) or []):
                            text = (w.word or "").strip()
                            if text:
                                words.append({"start": float(w.start) + offset,
                                              "end": float(w.end) + offset,
                                              "text": text,
                                              "prob": float(getattr(w, "probability", 1.0) or 1.0)})
                        recovered.append({
                            "start": float(seg.start) + offset,
                            "end": float(seg.end) + offset,
                            "text": (seg.text or "").strip(),
                            "words": words,
                            "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                            "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                            "recovered": True,
                        })
                except Exception as exc:  # noqa: BLE001
                    log.warning("gap fill failed for %.1f-%.1f: %s", a, b, exc)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if recovered:
            words = sum(len(s["words"]) for s in recovered)
            transcript.raw_segments.extend(recovered)
            transcript.raw_segments.sort(key=lambda s: s["start"])
            if on_log:
                on_log(f"Recovered {words} word(s) in {len(recovered)} segment(s) "
                       "Whisper had skipped")

    def _transcribe(
        self,
        audio_path: str,
        language: Optional[str],
        duration_hint: float,
        on_progress: Optional[Callable[[float, str], None]],
        on_log: Optional[Callable[[str], None]],
    ) -> Transcript:
        model = self.load(on_log=on_log)

        lang = None if (not language or language == "auto") else language
        vad_params = {"min_silence_duration_ms": int(self.cfg.get("vad_min_silence_ms", 500))}

        segments_iter, info = model.transcribe(
            audio_path,
            language=lang,
            beam_size=int(self.cfg.get("beam_size", 5)),
            vad_filter=bool(self.cfg.get("vad_filter", True)),
            vad_parameters=vad_params if self.cfg.get("vad_filter", True) else None,
            word_timestamps=bool(self.cfg.get("word_timestamps", True)),
            condition_on_previous_text=bool(self.cfg.get("condition_on_previous_text", False)),
        )

        total = duration_hint or getattr(info, "duration", 0.0) or 0.0
        detected = getattr(info, "language", None) or lang or "en"
        if on_log:
            prob = getattr(info, "language_probability", 0.0) or 0.0
            on_log(f"Detected language: {detected} ({prob * 100:.0f}% confidence)")

        raw: List[Dict[str, Any]] = []
        dropped = 0
        last_report = -1.0
        for seg in segments_iter:
            if _is_hallucination(seg):
                dropped += 1
                continue
            words = []
            for w in (getattr(seg, "words", None) or []):
                text = (w.word or "").strip()
                if not text:
                    continue
                words.append({
                    "start": float(w.start), "end": float(w.end),
                    "text": text, "prob": float(getattr(w, "probability", 1.0) or 1.0),
                })
            raw.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip(),
                "words": words,
                "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
            })
            if on_progress and total > 0 and seg.end - last_report > 5:
                last_report = seg.end
                on_progress(min(1.0, seg.end / total),
                            f"Transcribing… {int(seg.end // 60):02d}:{int(seg.end % 60):02d}"
                            f" / {int(total // 60):02d}:{int(total % 60):02d}")

        if on_progress:
            on_progress(1.0, "Transcription complete")
        if dropped and on_log:
            on_log(f"Discarded {dropped} non-speech segment(s) Whisper invented over "
                   "silence or background noise")

        end_time = raw[-1]["end"] if raw else total
        return Transcript(language=detected, duration=max(total, end_time), raw_segments=raw)


def words_from_segment(seg: Dict[str, Any]) -> List[Word]:
    return [Word(**w) for w in seg.get("words", [])]


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------
def _read_wav_mono16(path: str):
    """Read a 16-bit mono WAV into a numpy array. Whisper's input format."""
    import wave
    import numpy as np
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            return None, 0
        rate = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return data, rate


def _rms_db(samples, rate: int, start: float, end: float) -> float:
    import numpy as np
    a = max(0, int(start * rate))
    b = min(len(samples), int(end * rate))
    if b - a < rate // 10:
        return -120.0
    chunk = samples[a:b].astype("float32") / 32768.0
    rms = float(np.sqrt(np.mean(chunk * chunk))) or 1e-9
    return 20.0 * math.log10(rms)


def find_speech_gaps(raw: List[Dict[str, Any]], duration: float,
                     min_gap: float) -> List[tuple]:
    """Stretches with no transcript at all, long enough to hide real speech."""
    gaps: List[tuple] = []
    prev_end = 0.0
    for seg in sorted(raw, key=lambda s: s["start"]):
        if seg["start"] - prev_end >= min_gap:
            gaps.append((prev_end, seg["start"]))
        prev_end = max(prev_end, seg["end"])
    if duration - prev_end >= min_gap:
        gaps.append((prev_end, duration))
    return gaps


def _slice_wav(src: str, dst: str, start: float, end: float) -> bool:
    import wave
    with wave.open(src, "rb") as wf:
        rate = wf.getframerate()
        wf.setpos(min(wf.getnframes(), max(0, int(start * rate))))
        frames = wf.readframes(max(0, int((end - start) * rate)))
        params = wf.getparams()
    if not frames:
        return False
    with wave.open(dst, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return True


# Phrases Whisper famously emits over music, silence and background noise.
_HALLUCINATION_PHRASES = {
    "you", "oh", "thank you", "thanks", "thank you for watching",
    "thanks for watching", "please subscribe", "subscribe", "bye", "bye bye",
    "the end", "music", "applause", "silence", "outro",
    "merci", "merci d'avoir regardé", "sous-titres réalisés par la communauté d'amara.org",
    "abonnez-vous", "au revoir", "à bientôt",
}


def _normalise(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum() or ch.isspace()).strip()


def _is_hallucination(seg: Any) -> bool:
    """True for the junk Whisper produces when there is nothing to transcribe.

    Deliberately narrow: it needs the model to be *both* unsure this is speech
    *and* unsure of the words, on a very short line. Real speech, even quiet
    speech, does not satisfy all three at once.
    """
    text = (getattr(seg, "text", "") or "").strip()
    if not text:
        return True
    no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
    logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
    words = _normalise(text).split()

    if no_speech >= 0.6 and logprob < -0.5 and len(words) <= 3:
        return True
    # Bracketed sound tags are never dialogue: [Music], (applause), ♪...
    if text.startswith(("[", "(", "♪")) and text.endswith(("]", ")", "♪")):
        return True
    if no_speech >= 0.5 and _normalise(text) in _HALLUCINATION_PHRASES:
        return True
    return False
