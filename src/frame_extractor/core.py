"""Core frame-extraction logic (no CLI / MCP dependencies)."""
from __future__ import annotations

import csv
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("png", "jpg", "jpeg")
INDEX_FILENAME = "index.csv"
INDEX_COLUMNS = ("file", "frame_index", "seconds", "timestamp", "change_score")
SIGNATURE_WIDTH = 320  # width of the grayscale thumbnail used to score screen changes


class FrameExtractionError(Exception):
    """Raised for any user-facing failure during extraction."""


@dataclass
class FrameInfo:
    """One saved image (one row of ``index.csv``)."""

    file: str
    frame_index: int
    seconds: float
    change_score: Optional[float] = None  # vs. the previously saved frame; None for the first

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.seconds)


@dataclass
class ExtractionResult:
    video: str
    output_dir: str
    frames_saved: int
    frames_skipped: int  # unreadable / failed-to-write frames
    fps: float
    total_frames: int
    files: list[str]
    frames: list[FrameInfo] = field(default_factory=list)
    frames_unchanged: int = 0  # candidates dropped because change_score < min_change
    start_seconds: float = 0.0
    end_seconds: Optional[float] = None
    index_file: Optional[str] = None


def format_timestamp(seconds: float, sep: str = ":") -> str:
    """``3723.9`` -> ``"01:02:03"`` (fraction truncated); ``sep="-"`` gives a filename-safe form."""
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}{sep}{minutes:02d}{sep}{secs:02d}"


def parse_timestamp(value: str | float | int) -> float:
    """Parse ``90``, ``"90.5"``, ``"01:30"`` or ``"1:02:03.5"`` into seconds."""
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        error = ValueError(f"invalid time '{value}': use seconds or [hh:]mm:ss")
        parts = str(value).strip().split(":")
        if len(parts) > 3:
            raise error
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            raise error from None
        # Lower units must stay below 60 once a higher unit is present ("01:75" is a typo).
        if any(n < 0 for n in numbers) or any(n >= 60 for n in numbers[1:]):
            raise error
        seconds = 0.0
        for number in numbers:
            seconds = seconds * 60 + number
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"invalid time '{value}': must be a finite value >= 0")
    return seconds


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """Downscaled, slightly blurred grayscale copy of ``frame`` used to compare screens."""
    if frame.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        frame = cv2.cvtColor(frame, code)
    height, width = frame.shape[:2]
    if width > SIGNATURE_WIDTH:
        size = (SIGNATURE_WIDTH, max(1, round(height * SIGNATURE_WIDTH / width)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def change_score(signature: np.ndarray, reference: np.ndarray) -> float:
    """Mean absolute difference between two signatures, scaled to 0-100."""
    if signature.shape != reference.shape:
        reference = cv2.resize(reference, (signature.shape[1], signature.shape[0]),
                               interpolation=cv2.INTER_AREA)
    return float(cv2.absdiff(signature, reference).mean()) * 100.0 / 255.0


def save_image(target: Path, image: np.ndarray, jpeg_quality: int = 95) -> bool:
    """Encode ``image`` by the suffix of ``target`` and write it; False on failure."""
    ext = target.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext in (".jpg", ".jpeg") else []
    # imencode + write_bytes instead of imwrite: imwrite silently
    # fails on non-ASCII paths on Windows (e.g. "Padrão").
    try:
        ok, buf = cv2.imencode(ext, image, params)
        if ok:
            target.write_bytes(buf.tobytes())
        return bool(ok)
    except (cv2.error, OSError) as exc:
        logger.warning("Failed to write %s: %s", target, exc)
        return False


def load_image(path: str | Path) -> Optional[np.ndarray]:
    """Read an image (non-ASCII paths included); None if unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_index(path: str | Path, frames: list[FrameInfo]) -> Path:
    """Write ``frames`` as CSV (file names relative to the CSV's folder)."""
    target = Path(path)
    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(INDEX_COLUMNS)
        for f in frames:
            score = "" if f.change_score is None else f"{f.change_score:.2f}"
            writer.writerow([Path(f.file).name, f.frame_index, f"{f.seconds:.3f}", f.timestamp, score])
    return target


def read_index(path: str | Path) -> list[FrameInfo]:
    """Load an ``index.csv`` (or the one inside a frames folder) back into FrameInfo objects."""
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / INDEX_FILENAME
    if not source.is_file():
        raise FrameExtractionError(f"Index file not found: {source}")
    try:
        with open(source, newline="", encoding="utf-8") as fh:
            return [
                FrameInfo(
                    file=str(source.parent / row["file"]),
                    frame_index=int(row["frame_index"]),
                    seconds=float(row["seconds"]),
                    change_score=float(row["change_score"]) if row.get("change_score") else None,
                )
                for row in csv.DictReader(fh)
            ]
    except (KeyError, ValueError) as exc:
        raise FrameExtractionError(f"Malformed index file {source}: {exc}") from exc


def _open_capture(video: Path, scratch_dir: Path) -> tuple[cv2.VideoCapture, Optional[Path]]:
    """Open ``video``; returns the capture and a temporary alias to delete afterwards (or None).

    Some OpenCV builds/backends cannot open non-ASCII paths on Windows, so a
    failed open of such a path is retried through an ASCII-named hardlink
    (or copy) inside ``scratch_dir``.
    """
    cap = cv2.VideoCapture(str(video))
    if cap.isOpened():
        return cap, None
    cap.release()
    suffix = video.suffix.lower() if video.suffix.isascii() else ""
    alias = scratch_dir / f".frame_extractor_input{suffix}"
    if str(video).isascii() or not str(alias.resolve()).isascii():
        raise FrameExtractionError(f"OpenCV could not open the video: {video}")
    logger.info("Retrying %s through ASCII alias %s", video, alias)
    try:
        alias.unlink(missing_ok=True)
        try:
            os.link(video, alias)
        except OSError:
            shutil.copyfile(video, alias)  # different volume / no hardlink support
    except OSError as exc:
        raise FrameExtractionError(f"OpenCV could not open the video: {video} ({exc})") from exc
    cap = cv2.VideoCapture(str(alias))
    if cap.isOpened():
        return cap, alias
    cap.release()
    alias.unlink(missing_ok=True)
    raise FrameExtractionError(f"OpenCV could not open the video: {video}")


def _seek(cap: cv2.VideoCapture, target: int) -> int:
    """Position ``cap`` so the next grab() returns frame ``target``; returns that index."""
    if target <= 0:
        return 0
    if cap.set(cv2.CAP_PROP_POS_FRAMES, target):
        position = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if position > 0:
            return int(round(position))
    # Backend cannot seek: skip forward frame by frame (slow but exact).
    index = 0
    while index < target and cap.grab():
        index += 1
    return index


def _frame_seconds(cap: cv2.VideoCapture, index: int, fps: float) -> float:
    """Timestamp of the last grabbed frame: container PTS when available, else index / fps."""
    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if msec and msec > 0:
        return msec / 1000.0
    return index / fps if fps > 0 else 0.0


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_seconds: Optional[float] = None,
    every_frames: Optional[int] = None,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    *,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
    min_change: Optional[float] = None,
    timestamp_names: bool = False,
    write_index_file: bool = True,
) -> ExtractionResult:
    """Extract frames from a video and save them as images.

    At most one of ``every_seconds`` / ``every_frames`` may be given; if neither,
    every frame is extracted. ``progress_callback(current_frame, total_frames)``
    is called after each frame is read (used by the CLI's tqdm bar); both values
    are relative to the ``start_seconds``/``end_seconds`` window.

    ``start_seconds``/``end_seconds`` limit extraction to a time window (the
    sampling grid starts at ``start_seconds``). With ``min_change`` (0-100), a
    sampled frame is saved only if its :func:`change_score` against the last
    saved frame is >= ``min_change``; the first sampled frame is always saved.
    ``timestamp_names`` appends ``_hh-mm-ss`` to file names. An ``index.csv``
    (file, frame_index, seconds, timestamp, change_score) is written to the
    output folder unless ``write_index_file`` is False.
    """
    # ---- 1. Validate arguments -------------------------------------------
    video = Path(video_path).expanduser()
    if not video.is_file():
        raise FrameExtractionError(f"Video file not found: {video}")

    fmt = image_format.lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise FrameExtractionError(
            f"Unsupported format '{image_format}'. Use one of: png, jpg."
        )
    ext = "jpg" if fmt == "jpeg" else fmt

    if every_seconds is not None and every_frames is not None:
        raise FrameExtractionError("Use either every_seconds or every_frames, not both.")
    if every_seconds is not None and every_seconds <= 0:
        raise FrameExtractionError("every_seconds must be > 0.")
    if every_frames is not None and every_frames < 1:
        raise FrameExtractionError("every_frames must be >= 1.")
    if not 0 <= jpeg_quality <= 100:
        raise FrameExtractionError("jpeg_quality must be between 0 and 100.")
    if max_frames is not None and max_frames < 1:
        raise FrameExtractionError("max_frames must be >= 1.")
    for name, value in (("start_seconds", start_seconds), ("end_seconds", end_seconds)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise FrameExtractionError(f"{name} must be a finite value >= 0.")
    if end_seconds is not None and end_seconds <= (start_seconds or 0.0):
        raise FrameExtractionError("end_seconds must be greater than start_seconds.")
    if min_change is not None and not 0 <= min_change <= 100:
        raise FrameExtractionError("min_change must be between 0 and 100.")

    # ---- 2. Prepare output directory --------------------------------------
    out = Path(output_dir).expanduser()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FrameExtractionError(f"Cannot create output directory '{out}': {exc}") from exc

    # ---- 3. Open the video -------------------------------------------------
    cap, alias = _open_capture(video, out)

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Convert the interval into a frame step.
        if every_seconds is not None:
            if fps <= 0:
                raise FrameExtractionError("Video FPS unknown; use every_frames instead.")
            step = max(1, round(every_seconds * fps))
        else:
            step = every_frames or 1

        # Convert the time window into frame indexes.
        if (start_seconds or end_seconds is not None) and fps <= 0:
            raise FrameExtractionError("Video FPS unknown; start/end times cannot be used.")
        first = 0
        if start_seconds:
            first = round(start_seconds * fps)
            if total and first >= total:
                raise FrameExtractionError(
                    f"start ({format_timestamp(start_seconds)}) is past the end of the video "
                    f"({format_timestamp(total / fps)})."
                )
            first = _seek(cap, first)
        stop = total  # exclusive frame bound, only used for progress
        if end_seconds is not None:
            stop = math.floor(end_seconds * fps) + 1
            stop = min(stop, total) if total else stop
        span = max(0, stop - first)

        files: list[str] = []
        frames: list[FrameInfo] = []
        skipped = 0
        unchanged = 0
        index = first
        reference: Optional[np.ndarray] = None  # signature of the last saved frame

        # ---- 4. Read loop --------------------------------------------------
        while True:
            # grab() skips image conversion; retrieve() only for kept frames.
            if not cap.grab():
                break  # end of video (or unrecoverable read error)
            seconds = _frame_seconds(cap, index, fps)
            if end_seconds is not None and seconds > end_seconds + 1e-6:
                break
            if (index - first) % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    logger.warning("Unreadable frame %d skipped", index)
                    skipped += 1
                else:
                    signature = frame_signature(frame)
                    score = None if reference is None else change_score(signature, reference)
                    if min_change is not None and score is not None and score < min_change:
                        unchanged += 1
                    else:
                        stem = f"frame_{index:06d}"
                        if timestamp_names:
                            stem += f"_{format_timestamp(seconds, '-')}"
                        target = out / f"{stem}.{ext}"
                        if save_image(target, frame, jpeg_quality):
                            files.append(str(target))
                            frames.append(FrameInfo(str(target), index, seconds, score))
                            reference = signature
                        else:
                            skipped += 1
            index += 1
            if progress_callback:
                progress_callback(index - first, span)
            if max_frames is not None and len(files) >= max_frames:
                break
    finally:
        cap.release()
        if alias is not None:
            alias.unlink(missing_ok=True)

    if not files:
        if start_seconds or end_seconds is not None:
            raise FrameExtractionError("No frames were extracted in the requested time window.")
        raise FrameExtractionError("No frames were extracted (video empty or unreadable).")

    index_file = None
    if write_index_file:
        try:
            index_file = str(write_index(out / INDEX_FILENAME, frames))
        except OSError as exc:
            logger.warning("Failed to write %s: %s", out / INDEX_FILENAME, exc)

    return ExtractionResult(
        video=str(video),
        output_dir=str(out),
        frames_saved=len(files),
        frames_skipped=skipped,
        fps=fps,
        total_frames=total,
        files=files,
        frames=frames,
        frames_unchanged=unchanged,
        start_seconds=start_seconds or 0.0,
        end_seconds=end_seconds,
        index_file=index_file,
    )


def extract_frames_from_dir(
    videos_dir: str | Path,
    output_dir: str | Path,
    every_seconds: Optional[float] = 3.0,
    every_frames: Optional[int] = None,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    max_frames: Optional[int] = None,
    *,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
    min_change: Optional[float] = None,
    timestamp_names: bool = False,
    write_index_file: bool = True,
) -> list[ExtractionResult]:
    """Extract frames for all .mp4 files inside ``videos_dir`` into subfolders of ``output_dir``."""
    folder = Path(videos_dir).expanduser()
    if not folder.is_dir():
        raise FrameExtractionError(f"Videos directory not found: {folder}")

    videos = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    if not videos:
        raise FrameExtractionError(f"No .mp4 files found in directory: {folder}")

    out_root = Path(output_dir).expanduser()
    results: list[ExtractionResult] = []
    for video in videos:
        target_dir = out_root / video.stem
        res = extract_frames(
            video_path=video,
            output_dir=target_dir,
            every_seconds=every_seconds if every_frames is None else None,
            every_frames=every_frames,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            max_frames=max_frames,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            min_change=min_change,
            timestamp_names=timestamp_names,
            write_index_file=write_index_file,
        )
        results.append(res)
    return results

