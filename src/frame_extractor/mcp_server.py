"""MCP server exposing frame extraction as a tool (stdio transport)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .contact_sheet import build_contact_sheets, parse_grid
from .core import FrameExtractionError, extract_frames
from .paths import default_frames_dir

mcp = FastMCP("frame-extractor")


@mcp.tool()
def extract_video_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    every_seconds: Optional[float] = None,
    every_frames: Optional[int] = None,
    image_format: str = "jpg",
    max_frames: Optional[int] = 100,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
    min_change: Optional[float] = None,
    timestamp_names: bool = False,
    jpeg_quality: int = 95,
    contact_sheet: bool = False,
    sheet_grid: str = "3x3",
    contact_sheet_dir: Optional[str] = None,
) -> dict:
    """Extract frames from a video file on the host machine and save them as images.

    Args:
        video_path: Absolute path to the input video (e.g. an .mp4).
        output_dir: Directory where images will be written (created if missing). Default: a
            folder named after the video in the user's Pictures folder
            (<Pictures>/frame-extractor/<video name>).
        every_seconds: Save one frame every N seconds (mutually exclusive with every_frames).
        every_frames: Save one frame every N frames.
        image_format: "png" or "jpg".
        max_frames: Safety cap on saved images (default 100).
        start_seconds: Only extract from this video time on (seconds).
        end_seconds: Stop after this video time (seconds).
        min_change: 0-100; keep a sampled frame only if it differs from the last saved
            one by at least this score (mean abs. difference of a 320 px grayscale).
            About 2 removes repeated screens in screen-share recordings.
        timestamp_names: Name files frame_NNNNNN_hh-mm-ss instead of frame_NNNNNN.
        jpeg_quality: JPEG quality 0-100.
        contact_sheet: Also tile the saved frames into labelled contact sheets,
            so many frames can be reviewed with few image reads.
        sheet_grid: Contact sheet grid as COLSxROWS (default "3x3").
        contact_sheet_dir: Contact sheet folder (default <output_dir>/contact-sheets).
    """
    try:
        cols, rows = parse_grid(sheet_grid)
        r = extract_frames(
            video_path, output_dir or default_frames_dir(video_path),
            every_seconds=every_seconds, every_frames=every_frames,
            image_format=image_format, jpeg_quality=jpeg_quality, max_frames=max_frames,
            start_seconds=start_seconds, end_seconds=end_seconds,
            min_change=min_change, timestamp_names=timestamp_names,
        )
        sheets: list[str] = []
        if contact_sheet:
            sheet_dir = contact_sheet_dir or str(Path(r.output_dir) / "contact-sheets")
            sheets = build_contact_sheets(r.frames, sheet_dir, cols=cols, rows=rows)
    except (FrameExtractionError, ValueError) as exc:
        # Raising makes the MCP client receive an error result with this message.
        raise ValueError(str(exc)) from exc
    return {
        "output_dir": r.output_dir,
        "frames_saved": r.frames_saved,
        "frames_skipped": r.frames_skipped,
        "frames_unchanged": r.frames_unchanged,
        "fps": r.fps,
        "total_frames": r.total_frames,
        "index_file": r.index_file,
        "files": r.files[:50],  # keep the response small
        "timestamps": [f.timestamp for f in r.frames[:50]],
        "contact_sheets": sheets[:50],
    }


def main() -> None:
    # stdio transport: never print to stdout here, it carries the protocol.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
