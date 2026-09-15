"""Native "open file" dialog.

The browser cannot hand JavaScript a real filesystem path, and uploading a
3 GB podcast to a server running on the same machine would be absurd. Since the
server IS the local machine, we open a real Windows file picker from the backend
instead. Falls back to tkinter on non-Windows platforms.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

FILTERS = {
    "video": (
        "Video files\0*.mp4;*.mkv;*.mov;*.avi;*.webm;*.flv;*.wmv;*.m4v;*.ts;*.mpg;*.mpeg\0"
        "Audio files\0*.mp3;*.wav;*.m4a;*.aac;*.flac;*.ogg;*.opus\0"
        "All files\0*.*\0\0"
    ),
    "image": (
        "Image files\0*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif\0"
        "All files\0*.*\0\0"
    ),
    "audio": (
        "Audio files\0*.mp3;*.wav;*.m4a;*.aac;*.flac;*.ogg;*.opus;*.wma\0"
        "All files\0*.*\0\0"
    ),
}
TK_FILTERS = {
    "video": [("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v"),
              ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
              ("All files", "*.*")],
    "image": [("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
              ("All files", "*.*")],
    "audio": [("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus"),
              ("All files", "*.*")],
}
TITLES = {"video": "Select a video to analyse", "image": "Select a watermark image",
          "audio": "Select a background music track"}


def _win32_dialog(kind: str) -> Optional[str]:
    import ctypes
    from ctypes import wintypes

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", wintypes.LPVOID),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", wintypes.LPVOID),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    OFN_EXPLORER = 0x00080000
    OFN_FILEMUSTEXIST = 0x00001000
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_NOCHANGEDIR = 0x00000008

    buf = ctypes.create_unicode_buffer(4096)
    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter = FILTERS.get(kind, FILTERS["video"])
    ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
    ofn.nMaxFile = 4096
    ofn.lpstrTitle = TITLES.get(kind, TITLES["video"])
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR

    comdlg32 = ctypes.windll.comdlg32
    if comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        return buf.value or None
    return None


def _tk_dialog(kind: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title=TITLES.get(kind, TITLES["video"]),
            filetypes=TK_FILTERS.get(kind, TK_FILTERS["video"]),
        )
    finally:
        root.destroy()
    return path or None


def pick_file(timeout: float = 300.0, kind: str = "video") -> Optional[str]:
    """Open a modal picker on a worker thread and return the chosen path."""
    result: dict = {}

    def worker():
        try:
            result["path"] = _win32_dialog(kind) if os.name == "nt" else _tk_dialog(kind)
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result.get("path")
