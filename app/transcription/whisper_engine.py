"""faster-whisper transcription with word-level timestamps and VRAM hygiene."""
from __future__ import annotations

import gc
import logging
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
