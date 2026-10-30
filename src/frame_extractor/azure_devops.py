"""Azure DevOps -> frame extractor pipeline.

Downloads the video attachment(s) of a Work Item and feeds them into
``frame_extractor.core.extract_frames``.

REST endpoints used (api-version 7.1):
  GET {org}/_apis/wit/workitems/{id}?$expand=relations   -> work item + attachment links
  GET {org}/_apis/wit/attachments/{guid}?download=true    -> raw attachment bytes
Auth: HTTP Basic with empty user and the PAT as password.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from tqdm import tqdm

from .core import FrameExtractionError, extract_frames
from .paths import downloads_root, frames_root

API_VERSION = "7.1"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}
CHUNK = 1024 * 1024  # 1 MiB
MAX_RETRIES = 5


class AzureDevOpsError(Exception):
    """User-facing error for any Azure DevOps failure."""


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def make_session(pat: str) -> requests.Session:
    s = requests.Session()
    s.auth = ("", pat)  # Basic auth: blank username, PAT as password
    s.headers["Accept"] = "application/json"
    return s


def _check(resp: requests.Response, what: str) -> None:
    """Translate Azure DevOps HTTP responses into readable errors."""
    if resp.status_code in (401, 403):
        raise AzureDevOpsError(f"Unauthorized ({resp.status_code}) while {what}: "
                               "PAT invalid/expired or missing 'Work Items (Read)' scope.")
    # An invalid PAT sometimes yields 203 + an HTML sign-in page instead of 401.
    if resp.status_code == 203 or "text/html" in resp.headers.get("Content-Type", ""):
        raise AzureDevOpsError(f"Authentication failed while {what} (got sign-in page). Check the PAT.")
    if resp.status_code == 404:
        raise AzureDevOpsError(f"Not found while {what}. Check the Work Item ID and organization URL.")
    if not resp.ok:
        try:
            msg = resp.json().get("message", resp.text)
        except ValueError:
            msg = resp.text
        raise AzureDevOpsError(f"HTTP {resp.status_code} while {what}: {msg[:300]}")


# --------------------------------------------------------------------------
# API calls
# --------------------------------------------------------------------------
def get_video_attachments(session: requests.Session, org_url: str, work_item_id: int) -> list[dict]:
    """Return [{'name': ..., 'url': ..., 'size': ...}] for video attachments of a work item."""
    url = f"{org_url.rstrip('/')}/_apis/wit/workitems/{work_item_id}"
    try:
        resp = session.get(url, params={"$expand": "relations", "api-version": API_VERSION}, timeout=30)
    except requests.RequestException as exc:
        raise AzureDevOpsError(f"Network error fetching work item {work_item_id}: {exc}") from exc
    _check(resp, f"fetching work item {work_item_id}")

    videos = []
    # Attachments appear as relations with rel == "AttachedFile".
    for rel in resp.json().get("relations") or []:
        if rel.get("rel") != "AttachedFile":
            continue
        attrs = rel.get("attributes") or {}
        name = attrs.get("name") or unquote(Path(urlparse(rel["url"]).path).name)
        if Path(name).suffix.lower() in VIDEO_EXTENSIONS:
            videos.append({"name": name, "url": rel["url"], "size": attrs.get("resourceSize")})
    return videos


def download_attachment(session: requests.Session, url: str, dest: Path) -> Path:
    """Stream an attachment to ``dest`` with a tqdm bar.

    On network interruption, retries with exponential backoff and resumes via
    an HTTP Range header when the server supports it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    params = {"download": "true", "api-version": API_VERSION}
    bar = None

    for attempt in range(1, MAX_RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Accept": "application/octet-stream"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with session.get(url, params=params, headers=headers, stream=True, timeout=(15, 60)) as r:
                if r.status_code == 416:  # range past end: already complete
                    break
                _check(r, f"downloading {dest.name}")
                if have and r.status_code != 206:
                    have = 0  # server ignored Range: restart from scratch
                total = int(r.headers.get("Content-Length", 0)) + have or None
                if bar is None:
                    bar = tqdm(total=total, initial=have, unit="B", unit_scale=True,
                               unit_divisor=1024, desc=f"Downloading {dest.name}")
                else:
                    bar.reset(total=total)
                    bar.update(have)
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
                        bar.update(len(chunk))
            break  # success
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            if attempt == MAX_RETRIES:
                if bar:
                    bar.close()
                raise AzureDevOpsError(f"Download failed after {MAX_RETRIES} attempts: {exc}") from exc
            wait = 2 ** attempt
            tqdm.write(f"Network error ({exc.__class__.__name__}); retrying in {wait}s "
                       f"[{attempt}/{MAX_RETRIES}]...")
            time.sleep(wait)
    if bar:
        bar.close()
    part.replace(dest)
    return dest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ado-frames",
        description="Download video attachments from an Azure DevOps Work Item and extract frames.",
    )
    p.add_argument("work_item_id", type=int, help="Work Item ID (e.g. 402221)")
    p.add_argument("--org-url", default=os.environ.get("AZURE_DEVOPS_ORG_URL"),
                   help="Organization URL, e.g. https://dev.azure.com/MyOrg "
                        "(default: $AZURE_DEVOPS_ORG_URL)")
    p.add_argument("--pat", help="Personal Access Token (prefer env var AZURE_DEVOPS_PAT)")
    p.add_argument("--downloads-dir",
                   help="Where videos are saved (default: <Downloads>/frame-extractor)")
    p.add_argument("-o", "--output-dir",
                   help="Root folder for frames; one subfolder per video "
                        "(default: <Pictures>/frame-extractor)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-s", "--every-seconds", type=float, help="Save 1 frame every N seconds")
    g.add_argument("-n", "--every-frames", type=int, help="Save 1 frame every N frames")
    p.add_argument("-f", "--format", default="jpg", choices=["png", "jpg", "jpeg"])
    p.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality 0-100")
    p.add_argument("-m", "--max-frames", type=int, help="Stop after N images per video")
    p.add_argument("--all", action="store_true", help="Process every video attachment (default: first only)")
    p.add_argument("--skip-existing", action="store_true", help="Reuse already-downloaded files")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Env var wins so the token doesn't land in shell history / process list.
    pat = os.environ.get("AZURE_DEVOPS_PAT") or args.pat
    if not pat:
        print("Error: provide a PAT via AZURE_DEVOPS_PAT or --pat.", file=sys.stderr)
        return 2
    if not args.org_url:
        print("Error: provide --org-url or AZURE_DEVOPS_ORG_URL.", file=sys.stderr)
        return 2

    session = make_session(pat)
    downloads_dir = Path(args.downloads_dir).expanduser() if args.downloads_dir else downloads_root()
    frames_dir = Path(args.output_dir).expanduser() if args.output_dir else frames_root()
    try:
        videos = get_video_attachments(session, args.org_url, args.work_item_id)
        if not videos:
            print(f"Work Item {args.work_item_id} has no video attachments "
                  f"({', '.join(sorted(VIDEO_EXTENSIONS))}).", file=sys.stderr)
            return 3
        print(f"Found {len(videos)} video attachment(s): " + ", ".join(v["name"] for v in videos))
        if not args.all:
            videos = videos[:1]

        for v in videos:
            # Prefix with the WI id to avoid collisions between work items.
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in v["name"])
            dest = downloads_dir / f"{args.work_item_id}_{safe}"
            if args.skip_existing and dest.exists():
                print(f"Using existing {dest}")
            else:
                download_attachment(session, v["url"], dest)
                print(f"Downloaded {dest}")

            # Handoff to the frame extractor (imported, no subprocess).
            out = frames_dir / dest.stem
            with tqdm(unit="frame", desc=f"Extracting {dest.name}") as bar:
                def on_progress(cur: int, total: int) -> None:
                    if total and bar.total != total:
                        bar.total = total
                    bar.n = cur
                    bar.refresh()

                result = extract_frames(
                    dest, out,
                    every_seconds=args.every_seconds, every_frames=args.every_frames,
                    image_format=args.format, jpeg_quality=args.quality,
                    max_frames=args.max_frames, progress_callback=on_progress,
                )
            print(f"Saved {result.frames_saved} frames to {result.output_dir} "
                  f"({result.frames_skipped} skipped)")
    except (AzureDevOpsError, FrameExtractionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Filesystem error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
