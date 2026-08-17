"""All FFmpeg / FFprobe interaction: probing, audio extraction, clip cutting."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ..models import VideoInfo

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v",
              ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".ogv", ".3gp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class FFmpegError(RuntimeError):
    pass


class FFmpeg:
    def __init__(self, ffmpeg_path: str = "", ffprobe_path: str = ""):
        self.ffmpeg = self._locate(ffmpeg_path, "ffmpeg")
        self.ffprobe = self._locate(ffprobe_path, "ffprobe")

    # ------------------------------------------------------------------ util
    @staticmethod
    def _locate(explicit: str, name: str) -> str:
        if explicit:
            p = Path(explicit)
            if p.is_dir():
                cand = p / (name + (".exe" if os.name == "nt" else ""))
                if cand.exists():
                    return str(cand)
            elif p.exists():
                return str(p)
        found = shutil.which(name)
        if found:
            return found
        # Common Windows install locations. winget in particular unpacks into
        # ...\WinGet\Packages\Gyan.FFmpeg_.../ffmpeg-*-full_build/bin and only
        # adds that to PATH after the shell is restarted.
        if os.name == "nt":
            local = Path(os.environ.get("LOCALAPPDATA", ""))
            for base in (
                local / "Microsoft/WinGet/Links",
                Path("C:/ffmpeg/bin"),
                Path("C:/Program Files/ffmpeg/bin"),
                Path(os.environ.get("ProgramData", "")) / "chocolatey/bin",
            ):
                cand = base / f"{name}.exe"
                if cand.exists():
                    return str(cand)
            packages = local / "Microsoft/WinGet/Packages"
            if packages.is_dir():
                for pkg in sorted(packages.glob("*FFmpeg*"), reverse=True):
                    for cand in sorted(pkg.glob(f"**/bin/{name}.exe"), reverse=True):
                        return str(cand)
        return name  # let it fail loudly later with a clear message

    def available(self) -> bool:
        try:
            subprocess.run([self.ffmpeg, "-version"], capture_output=True,
                           creationflags=_NO_WINDOW, timeout=15)
            return True
        except Exception:
            return False

    def _run(self, args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run(args, capture_output=True, creationflags=_NO_WINDOW,
                                  timeout=timeout)
        except FileNotFoundError as exc:
            raise FFmpegError(
                "FFmpeg was not found. Install it (winget install Gyan.FFmpeg) or set "
                "general.ffmpeg_path in config.yaml."
            ) from exc
        return proc

    # ----------------------------------------------------------------- probe
    def probe(self, path: str) -> VideoInfo:
        src = Path(path)
        if not src.exists():
            raise FFmpegError(f"File not found: {path}")

        proc = self._run([
            self.ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(src),
        ], timeout=90)
        if proc.returncode != 0:
            raise FFmpegError(
                f"ffprobe failed for {src.name}: {proc.stderr.decode('utf-8', 'ignore')[:400]}"
            )
        data = json.loads(proc.stdout.decode("utf-8", "ignore") or "{}")

        streams = data.get("streams", [])
        vstream = next((s for s in streams if s.get("codec_type") == "video"
                        and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
        astream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = 0.0
        for source in (data.get("format", {}), vstream or {}, astream or {}):
            try:
                duration = max(duration, float(source.get("duration") or 0.0))
            except (TypeError, ValueError):
                pass

        fps = 0.0
        if vstream:
            for key in ("avg_frame_rate", "r_frame_rate"):
                raw = vstream.get(key) or "0/0"
                try:
                    num, _, den = raw.partition("/")
                    if float(den or 0) > 0:
                        fps = float(num) / float(den)
                        break
                except (TypeError, ValueError):
                    continue

        return VideoInfo(
            path=str(src),
            filename=src.name,
            duration=duration,
            width=int(vstream.get("width", 0)) if vstream else 0,
            height=int(vstream.get("height", 0)) if vstream else 0,
            fps=fps,
            size_bytes=src.stat().st_size,
            vcodec=(vstream or {}).get("codec_name", ""),
            acodec=(astream or {}).get("codec_name", ""),
            has_audio=astream is not None,
        )

    # --------------------------------------------------------- audio extract
    def extract_audio(self, path: str, out_path: Optional[str] = None,
                      sample_rate: int = 16000) -> str:
        """Decode to 16 kHz mono WAV - exactly what Whisper wants. Streams to disk."""
        if out_path is None:
            fd, out_path = tempfile.mkstemp(prefix="clipfinder_", suffix=".wav")
            os.close(fd)
        args = [
            self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(sample_rate),
            "-acodec", "pcm_s16le",
            "-map", "0:a:0?",
            str(out_path),
        ]
        proc = self._run(args)
        if proc.returncode != 0 or not Path(out_path).exists() or Path(out_path).stat().st_size < 1024:
            raise FFmpegError(
                "Audio extraction failed. Does the file contain an audio track?\n"
                + proc.stderr.decode("utf-8", "ignore")[-500:]
            )
        return str(out_path)

    # ------------------------------------------------------------ clip cutting
    def cut(self, src: str, out: str, start: float, end: float, *,
            mode: str = "copy", crf: int = 20, preset: str = "veryfast") -> str:
        start = max(0.0, start)
        duration = max(0.2, end - start)
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        if mode == "copy":
            args = [
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart", str(out),
            ]
        else:
            args = [
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", str(out),
            ]
        proc = self._run(args)
        if proc.returncode != 0:
            # stream copy can fail on odd containers - fall back to re-encode once
            if mode == "copy":
                return self.cut(src, out, start, end, mode="precise", crf=crf, preset=preset)
            raise FFmpegError(proc.stderr.decode("utf-8", "ignore")[-500:])
        return str(out)

    # ------------------------------------------------------ caption burning
    def has_encoder(self, name: str) -> bool:
        try:
            proc = self._run([self.ffmpeg, "-hide_banner", "-encoders"], timeout=20)
            return name.encode() in proc.stdout
        except Exception:  # noqa: BLE001
            return False

    def pick_encoder(self, preference: str = "auto") -> List[str]:
        """NVENC when available - burning captions re-encodes, and on an RTX
        card that is several times faster than libx264."""
        pref = (preference or "auto").lower()
        if pref in ("nvenc", "h264_nvenc") or (pref == "auto" and self.has_encoder("h264_nvenc")):
            if self.has_encoder("h264_nvenc"):
                return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23",
                        "-b:v", "0", "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]

    def burn(self, src: str, out: str, start: float, end: float, ass_path: str, *,
             encoder: str = "auto", fonts_dir: Optional[str] = None) -> str:
        """Cut [start, end] and burn an ASS subtitle file into the picture.

        The subtitles filter parses its argument as a filtergraph token, where
        ':' and '\\' are syntax - a nightmare with Windows paths. Instead of
        escaping, we run FFmpeg with cwd set to the .ass file's folder and pass
        a bare ASCII filename, which has neither.
        """
        ass = Path(ass_path)
        duration = max(0.2, end - start)
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        vf = f"subtitles={ass.name}"
        if fonts_dir:
            fonts = Path(fonts_dir)
            if fonts.is_dir():
                # Relative to cwd, so it stays free of colons too.
                try:
                    rel = fonts.resolve().relative_to(ass.parent.resolve())
                    vf += f":fontsdir={rel.as_posix()}"
                except ValueError:
                    vf += f":fontsdir={fonts.name}"

        args = [
            self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}", "-i", str(Path(src).resolve()),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            *self.pick_encoder(encoder),
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(Path(out).resolve()),
        ]
        proc = subprocess.run(args, capture_output=True, creationflags=_NO_WINDOW,
                              cwd=str(ass.parent))
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "ignore")[-600:]
            if "h264_nvenc" in " ".join(args):
                # NVENC can fail if the GPU is busy or the driver refuses the
                # session; CPU encoding always works.
                return self.burn(src, out, start, end, ass_path,
                                 encoder="libx264", fonts_dir=fonts_dir)
            raise FFmpegError(f"Caption burn failed: {err}")
        return str(out)

    def preview_stream(self, src: str, start: float, end: float,
                       height: int = 720) -> Iterable[bytes]:
        """Transcode a section on the fly to fragmented MP4 for in-browser preview.

        Used as a fallback when the browser cannot decode the source container
        (mkv, av1, hevc...).
        """
        duration = max(0.5, end - start)
        args = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}", "-i", str(src), "-t", f"{duration:.3f}",
            "-vf", f"scale=-2:{height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4", "pipe:1",
        ]
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=_NO_WINDOW)
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)

    def thumbnail(self, src: str, at: float, out: str, width: int = 480) -> Optional[str]:
        args = [
            self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, at):.3f}", "-i", str(src), "-frames:v", "1",
            "-vf", f"scale={width}:-2", str(out),
        ]
        proc = self._run(args, timeout=60)
        return str(out) if proc.returncode == 0 and Path(out).exists() else None


def is_media_file(path: str) -> bool:
    return Path(path).suffix.lower() in (VIDEO_EXTS | AUDIO_EXTS)
