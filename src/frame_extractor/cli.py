"""Command line interface: ``frame-extractor video.mp4 -o frames --every-seconds 2``."""
from __future__ import annotations

import argparse
import sys

from pathlib import Path

from tqdm import tqdm

from .contact_sheet import build_contact_sheets, parse_grid
from .core import (
    ExtractionResult,
    FrameExtractionError,
    extract_frames,
    extract_frames_from_dir,
    parse_timestamp,
)
from .paths import default_frames_dir, frames_root


def _time_arg(value: str) -> float:
    try:
        return parse_timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _grid_arg(value: str) -> tuple[int, int]:
    try:
        return parse_grid(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frame-extractor", description="Extract frames from a video or directory of videos as images."
    )
    p.add_argument("video", help="Path to the input video (.mp4) or directory containing .mp4 files")
    p.add_argument("-o", "--output-dir",
                   help="Output directory (default: <Pictures>/frame-extractor/<video name>; "
                        "in directory mode, one subfolder per video under <Pictures>/frame-extractor)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("-s", "--every-seconds", type=float, help="Save 1 frame every N seconds")
    group.add_argument("-n", "--every-frames", type=int, help="Save 1 frame every N frames")
    p.add_argument("-f", "--format", default="jpg", choices=["png", "jpg", "jpeg"],
                   help="Output image format (default: jpg)")
    p.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality 0-100 (default: 95)")
    p.add_argument("-m", "--max-frames", type=int, help="Stop after saving N images")
    p.add_argument("--start", type=_time_arg, metavar="TIME",
                   help="Start time, seconds or [hh:]mm:ss (default: beginning of the video)")
    p.add_argument("--end", type=_time_arg, metavar="TIME",
                   help="End time, seconds or [hh:]mm:ss (default: end of the video)")
    p.add_argument("--min-change", type=float, metavar="SCORE",
                   help="Skip sampled frames whose change score (0-100, mean abs. difference of a "
                        "320 px grayscale) vs. the last saved frame is below SCORE; ~2 for screen shares")
    p.add_argument("--timestamps", action="store_true",
                   help="Add the video time to file names: frame_000123_00-00-07.jpg")
    p.add_argument("--no-index", action="store_true",
                   help="Do not write index.csv (file, frame_index, seconds, timestamp, change_score)")
    p.add_argument("--contact-sheet", action="store_true",
                   help="Also tile the saved frames into labelled JPEG contact sheets")
    p.add_argument("--sheet-grid", type=_grid_arg, default=(3, 3), metavar="COLSxROWS",
                   help="Contact sheet grid (default: 3x3)")
    p.add_argument("--sheet-width", type=int, default=640, metavar="PX",
                   help="Thumbnail width inside contact sheets (default: 640)")
    p.add_argument("--sheet-dir",
                   help="Contact sheet folder (default: <output-dir>/contact-sheets; "
                        "one subfolder per video in directory mode)")
    return p


def _contact_sheets(args: argparse.Namespace, result: ExtractionResult, sheet_dir: Path) -> None:
    cols, rows = args.sheet_grid
    sheets = build_contact_sheets(result.frames, sheet_dir, cols=cols, rows=rows,
                                  thumb_width=args.sheet_width, jpeg_quality=args.quality)
    noun = "contact sheet" if len(sheets) == 1 else "contact sheets"
    print(f"Built {len(sheets)} {noun} ({cols}x{rows}) in {sheet_dir}")


def _summary(result: ExtractionResult, min_change: float | None) -> str:
    detail = f"{result.frames_skipped} skipped"
    if min_change is not None:
        detail += f", {result.frames_unchanged} unchanged dropped"
    return detail


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = Path(args.video).expanduser()
    window = dict(start_seconds=args.start, end_seconds=args.end, min_change=args.min_change,
                  timestamp_names=args.timestamps, write_index_file=not args.no_index)

    if target.is_dir():
        try:
            results = extract_frames_from_dir(
                target,
                args.output_dir or frames_root(),
                every_seconds=args.every_seconds if args.every_seconds is not None else (None if args.every_frames else 3.0),
                every_frames=args.every_frames,
                image_format=args.format,
                jpeg_quality=args.quality,
                max_frames=args.max_frames,
                **window,
            )
            total_saved = sum(r.frames_saved for r in results)
            for r in results:
                print(f"[OK] {Path(r.video).name} -> {r.frames_saved} frames ({r.output_dir})")
                if args.contact_sheet:
                    sheet_dir = (Path(args.sheet_dir) / Path(r.video).stem if args.sheet_dir
                                 else Path(r.output_dir) / "contact-sheets")
                    _contact_sheets(args, r, sheet_dir)
        except FrameExtractionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130
        print(f"Batch complete: {len(results)} videos, {total_saved} total frames saved.")
        return 0

    bar = tqdm(unit="frame", desc="Extracting")

    def on_progress(current: int, total: int) -> None:
        # Total is only known after the video is opened, so set it lazily.
        if total and bar.total != total:
            bar.total = total
        bar.update(current - bar.n)  # update() honours mininterval, unlike refresh()

    try:
        result = extract_frames(
            args.video, args.output_dir or default_frames_dir(target),
            every_seconds=args.every_seconds, every_frames=args.every_frames,
            image_format=args.format, jpeg_quality=args.quality,
            max_frames=args.max_frames, progress_callback=on_progress,
            **window,
        )
    except FrameExtractionError as exc:
        bar.close()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        bar.close()
        print("Interrupted.", file=sys.stderr)
        return 130
    bar.close()
    print(f"Saved {result.frames_saved} frames to {result.output_dir} "
          f"({_summary(result, args.min_change)})")
    if args.contact_sheet:
        sheet_dir = Path(args.sheet_dir) if args.sheet_dir else Path(result.output_dir) / "contact-sheets"
        try:
            _contact_sheets(args, result, sheet_dir)
        except FrameExtractionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
