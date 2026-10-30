"""Contact sheets: tile many frames, each labelled with its timestamp, into a few review images."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Optional, Union

import cv2
import numpy as np

from .core import FrameExtractionError, FrameInfo, format_timestamp, load_image, save_image

FrameLike = Union[FrameInfo, str, Path]

_NAME_RE = re.compile(r"frame_(\d+)(?:_(\d{2,})-(\d{2})-(\d{2}))?")
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PAD = 8
_BACKGROUND = (40, 40, 40)


def parse_grid(spec: str) -> tuple[int, int]:
    """``"3x3"`` -> ``(cols, rows)``."""
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", spec or "")
    if not match or int(match.group(1)) < 1 or int(match.group(2)) < 1:
        raise ValueError(f"invalid grid '{spec}': use COLSxROWS, e.g. 3x3")
    return int(match.group(1)), int(match.group(2))


def _describe(item: FrameLike) -> tuple[Path, str, Optional[float]]:
    """(image path, label, seconds or None) for a FrameInfo or a frame file path."""
    if isinstance(item, FrameInfo):
        label = f"{item.timestamp}  #{item.frame_index}"
        if item.change_score is not None:
            label += f"  chg {item.change_score:.1f}"
        return Path(item.file), label, item.seconds
    path = Path(item)
    match = _NAME_RE.search(path.stem)
    if match and match.group(2):
        h, m, s = (int(g) for g in match.group(2, 3, 4))
        return path, f"{h:02d}:{m:02d}:{s:02d}  #{int(match.group(1))}", float(h * 3600 + m * 60 + s)
    return path, path.stem, None


def _thumbnail(path: Path, width: int) -> np.ndarray:
    image = load_image(path)
    if image is None:
        image = np.zeros((round(width * 9 / 16), width, 3), np.uint8)
        cv2.putText(image, "unreadable", (10, 40), _FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return image
    height, src_width = image.shape[:2]
    interpolation = cv2.INTER_AREA if src_width > width else cv2.INTER_LINEAR
    return cv2.resize(image, (width, max(1, round(height * width / src_width))), interpolation=interpolation)


def _fit(label: str, max_width: int, scale: float, thickness: int) -> str:
    # Hershey fonts only render ASCII; trim the label until it fits the tile.
    text = label.encode("ascii", "replace").decode("ascii")
    while len(text) > 1 and cv2.getTextSize(text, _FONT, scale, thickness)[0][0] > max_width:
        text = text[:-1]
    return text


def build_contact_sheets(
    frames: Iterable[FrameLike],
    out_dir: str | Path,
    cols: int = 3,
    rows: int = 3,
    thumb_width: int = 640,
    jpeg_quality: int = 85,
    prefix: str = "sheet",
) -> list[str]:
    """Tile ``frames`` into ``cols`` x ``rows`` grids with a timestamp label per tile.

    ``frames`` may be :class:`FrameInfo` objects (``ExtractionResult.frames`` or
    ``read_index(folder)``) or image paths, whose ``frame_NNNNNN_hh-mm-ss`` names
    give the labels. Sheets are written as ``<prefix>_NNN_<first>_<last>.jpg``
    (time range of the tiles) and their paths are returned in order.
    """
    if cols < 1 or rows < 1:
        raise FrameExtractionError("Contact sheet cols and rows must be >= 1.")
    if thumb_width < 64:
        raise FrameExtractionError("thumb_width must be >= 64.")
    if not 0 <= jpeg_quality <= 100:
        raise FrameExtractionError("jpeg_quality must be between 0 and 100.")
    items = [_describe(f) for f in frames]
    if not items:
        raise FrameExtractionError("No frames to build contact sheets from.")

    out = Path(out_dir).expanduser()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FrameExtractionError(f"Cannot create contact sheet directory '{out}': {exc}") from exc

    scale = 0.7 * thumb_width / 640
    thickness = max(1, round(thumb_width / 400))
    (_, text_height), baseline = cv2.getTextSize("00:00:00", _FONT, scale, thickness)
    band = text_height + baseline + 10  # black label strip above each thumbnail

    per_sheet = cols * rows
    sheets: list[str] = []
    for number, first in enumerate(range(0, len(items), per_sheet), start=1):
        chunk = items[first:first + per_sheet]
        thumbs = [_thumbnail(path, thumb_width) for path, _, _ in chunk]
        tile_height = band + max(t.shape[0] for t in thumbs)
        used_rows = math.ceil(len(chunk) / cols)
        canvas = np.full(
            (_PAD + used_rows * (tile_height + _PAD), _PAD + cols * (thumb_width + _PAD), 3),
            _BACKGROUND, dtype=np.uint8,
        )
        for position, ((_, label, _), thumb) in enumerate(zip(chunk, thumbs)):
            row, col = divmod(position, cols)
            x = _PAD + col * (thumb_width + _PAD)
            y = _PAD + row * (tile_height + _PAD)
            canvas[y:y + band, x:x + thumb_width] = 0
            text = _fit(label, thumb_width - 12, scale, thickness)
            cv2.putText(canvas, text, (x + 6, y + band - baseline - 5), _FONT, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            canvas[y + band:y + band + thumb.shape[0], x:x + thumb.shape[1]] = thumb

        name = f"{prefix}_{number:03d}"
        start_s, end_s = chunk[0][2], chunk[-1][2]
        if start_s is not None and end_s is not None:
            name += f"_{format_timestamp(start_s, '-')}_{format_timestamp(end_s, '-')}"
        target = out / f"{name}.jpg"
        if not save_image(target, canvas, jpeg_quality):
            raise FrameExtractionError(f"Failed to write contact sheet {target}")
        sheets.append(str(target))
    return sheets
