"""Make the pip-installed CUDA runtime DLLs visible to CTranslate2 on Windows.

faster-whisper needs cuBLAS and cuDNN 9. Rather than requiring a full CUDA
Toolkit install we ship them through the `nvidia-cublas-cu12` /
`nvidia-cudnn-cu12` wheels and point the loader at their `bin` folders.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

_DONE = False


def _candidate_dirs() -> List[Path]:
    dirs: List[Path] = []
    for site in sys.path:
        base = Path(site) / "nvidia"
        if not base.is_dir():
            continue
        for pkg in ("cublas", "cudnn", "cuda_runtime"):
            for sub in ("bin", "lib"):
                d = base / pkg / sub
                if d.is_dir():
                    dirs.append(d)
    return dirs


def ensure_cuda_dlls() -> List[str]:
    """Idempotently add the NVIDIA wheel DLL folders to the search path."""
    global _DONE
    added: List[str] = []
    if _DONE or os.name != "nt":
        return added
    for d in _candidate_dirs():
        try:
            os.add_dll_directory(str(d))
            added.append(str(d))
        except (OSError, AttributeError):
            continue
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    _DONE = True
    return added


def cuda_available() -> bool:
    """True if CTranslate2 reports at least one usable CUDA device."""
    ensure_cuda_dlls()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
