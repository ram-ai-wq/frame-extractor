"""Default output folders in the user's standard OS locations (Windows, macOS, Linux).

Outputs never land in the project or the current directory unless the caller asks for it
(``-o``/``--downloads-dir`` or an explicit ``output_dir``). platformdirs resolves the real
folders: Windows known folders (also when redirected, e.g. to OneDrive), the macOS home
folders and the Linux XDG user directories (``~/.config/user-dirs.dirs``).
"""
from __future__ import annotations

from pathlib import Path

import platformdirs

APP_NAME = "frame-extractor"


def frames_root() -> Path:
    """``<Pictures>/frame-extractor``: one subfolder of frames per video."""
    return Path(platformdirs.user_pictures_dir()) / APP_NAME


def downloads_root() -> Path:
    """``<Downloads>/frame-extractor``: videos downloaded by ``ado-frames``."""
    return Path(platformdirs.user_downloads_dir()) / APP_NAME


def transcripts_root() -> Path:
    """``<Documents>/frame-extractor/transcripts``: ``video-transcribe`` outputs."""
    return Path(platformdirs.user_documents_dir()) / APP_NAME / "transcripts"


def default_frames_dir(video: str | Path) -> Path:
    """Frames folder used when no output folder is given: ``<Pictures>/frame-extractor/<video name>``."""
    return frames_root() / Path(video).stem
