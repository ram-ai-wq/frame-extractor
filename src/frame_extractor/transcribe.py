"""Speech-to-text for video/audio files with faster-whisper (CLI: ``video-transcribe``).

Pipeline: ffmpeg extracts a 16 kHz mono WAV (optionally only a ``start``/``end``
window), the audio is split into chunks cut at the quietest point near each chunk
boundary (bounded memory, checkpointed outputs), and faster-whisper (CTranslate2,
CPU int8 by default) transcribes each chunk. Outputs ``<name>.txt`` (``[hh:mm:ss] text``),
``<name>.srt`` and ``<name>.json``; timestamps are always absolute to the original media.

Requires the optional extra: ``pip install -e ".[transcribe]"`` (faster-whisper) and
ffmpeg on PATH (or ``imageio-ffmpeg`` installed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .paths import transcripts_root

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = "pt"
DEFAULT_CHUNK_SECONDS = 300.0
MEMORY_RETRIES = 4  # attempts when MKL/CTranslate2 cannot allocate (busy machine, pagefile still growing)
MEMORY_RETRY_WAIT = 15.0  # seconds

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")
_MEMORY_ERROR_RE = re.compile(r"alloc|out of memory|bad_alloc", re.IGNORECASE)


class TranscriptionError(Exception):
    """Raised for any user-facing failure during transcription."""


class TranscriptionSetupError(TranscriptionError):
    """Missing prerequisite (ffmpeg, faster-whisper); the CLI exits with code 2."""


@dataclass
class TranscriptSegment:
    start: float  # seconds, absolute to the original media
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0

    @property
    def low_confidence(self) -> bool:
        """Likely misheard/hallucinated, per Whisper's fallback thresholds.

        faster-whisper reports these metrics per 30 s decoding window, so neighbouring segments share
        them. A high ``no_speech_prob`` only counts together with a mediocre ``avg_logprob``.
        """
        return (self.avg_logprob < -1.0 or self.compression_ratio > 2.4
                or (self.no_speech_prob > 0.6 and self.avg_logprob < -0.7))


@dataclass
class TranscriptionResult:
    media: str
    output_dir: str
    files: dict[str, str]
    model: str
    language: Optional[str]
    language_probability: float
    duration: Optional[float]  # whole media, seconds (None if ffprobe/ffmpeg could not tell)
    window_start: float
    window_end: float
    audio_duration: float  # seconds transcribed in this run
    speech_duration: float  # seconds kept by the VAD in this run
    segments: list[TranscriptSegment]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """Audio seconds transcribed per wall-clock second (model load and extraction excluded)."""
        spent = self.timings.get("transcribe", 0.0)
        return self.audio_duration / spent if spent > 0 else 0.0


# ---- time helpers ---------------------------------------------------------------

def parse_time(value: str | float | int | None) -> Optional[float]:
    """Parse seconds (``90``, ``90.5``) or ``[hh:]mm:ss[.ms]`` (``01:30``, ``1:02:03``) into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = value.strip().replace(",", ".")
        try:
            seconds = float(text)
        except ValueError:
            match = _TIME_RE.match(text)
            if not match:
                raise TranscriptionError(
                    f"Invalid time '{value}'. Use seconds (90 or 90.5) or [hh:]mm:ss (01:30, 1:02:03)."
                ) from None
            hours, minutes, secs = match.groups()
            if float(secs) >= 60 or (hours is not None and int(minutes) >= 60):
                raise TranscriptionError(f"Invalid time '{value}': minutes/seconds must be < 60.")
            seconds = int(hours or 0) * 3600 + int(minutes) * 60 + float(secs)
    if not math.isfinite(seconds) or seconds < 0:
        raise TranscriptionError(f"Invalid time '{value}': must be a finite number >= 0.")
    return seconds


def format_timestamp(seconds: float, millis: bool = False, separator: str = ".") -> str:
    """``hh:mm:ss`` (truncated) or ``hh:mm:ss<sep>mmm`` when ``millis`` is set."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base}{separator}{ms:03d}" if millis else base


# ---- ffmpeg ---------------------------------------------------------------------

def find_ffmpeg() -> str:
    """Return the ffmpeg executable: PATH first, then the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # not installed, or its binary is missing
        pass
    raise TranscriptionSetupError(
        "ffmpeg not found. Install it and put it on PATH (Windows: winget install Gyan.FFmpeg; "
        "macOS: brew install ffmpeg; Linux: sudo apt install ffmpeg or your distro's package) "
        "or run: pip install imageio-ffmpeg"
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    # List arguments go through CreateProcessW on Windows, so non-ASCII paths are safe.
    return subprocess.run(
        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def probe_duration(media: Path, ffmpeg: Optional[str] = None) -> Optional[float]:
    """Media duration in seconds via ffprobe (or ``ffmpeg -i`` output); ``None`` if unknown."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        proc = _run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(media)])
        try:
            return float(proc.stdout.strip())
        except ValueError:
            pass
    if ffmpeg:
        proc = _run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(media)])
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def extract_audio(media: Path, wav_path: Path, start: float = 0.0, end: Optional[float] = None,
                  ffmpeg: Optional[str] = None) -> Path:
    """Write the ``[start, end)`` window of the first audio stream as a 16 kHz mono PCM WAV."""
    exe = ffmpeg or find_ffmpeg()
    cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]  # input seek: fast, and sample-accurate when transcoding
    cmd += ["-i", str(media)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", str(wav_path)]
    proc = _run(cmd)
    if proc.returncode != 0 or not wav_path.is_file():
        detail = proc.stderr.strip().splitlines()[-3:]
        if "matches no streams" in proc.stderr:
            raise TranscriptionError(f"No audio stream in {media}")
        raise TranscriptionError(f"ffmpeg failed to extract audio from {media}: {' | '.join(detail)}")
    return wav_path


def read_wav(wav_path: Path) -> np.ndarray:
    """Load a 16 kHz mono 16-bit WAV as an int16 array (float conversion happens per chunk)."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            if (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) != (1, 2, SAMPLE_RATE):
                raise TranscriptionError(f"Unexpected WAV format in {wav_path} (need 16 kHz mono 16-bit).")
            return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    except (wave.Error, EOFError, OSError) as exc:
        raise TranscriptionError(f"Cannot read extracted audio {wav_path}: {exc}") from exc


# ---- chunking -------------------------------------------------------------------

def _quietest_point(samples: np.ndarray, target: int, search: int) -> int:
    """Sample index of the quietest 0.5 s stretch within ``target ± search``."""
    lo, hi = max(0, target - search), min(len(samples), target + search)
    hop = SAMPLE_RATE // 50  # 20 ms frames
    frames = (hi - lo) // hop
    kernel = 25  # 25 frames = 0.5 s, so cuts land inside pauses rather than between syllables
    if frames <= kernel:
        return target
    window = samples[lo:lo + frames * hop].astype(np.float32).reshape(frames, hop)
    energy = np.square(window).mean(axis=1)
    smooth = np.convolve(energy, np.full(kernel, 1.0 / kernel, dtype=np.float32), mode="valid")
    return lo + (int(np.argmin(smooth)) + kernel // 2) * hop + hop // 2


def plan_chunks(samples: np.ndarray, chunk_seconds: float, search_seconds: float = 15.0) -> list[tuple[int, int]]:
    """Split audio into ~``chunk_seconds`` pieces cut at pauses; ``chunk_seconds <= 0`` disables."""
    total = len(samples)
    step = int(chunk_seconds * SAMPLE_RATE)
    if step <= 0 or total <= step * 1.2:
        return [(0, total)]
    search = int(min(search_seconds, chunk_seconds / 4) * SAMPLE_RATE)
    bounds: list[tuple[int, int]] = []
    begin = 0
    while total - begin > step * 1.2:
        cut = _quietest_point(samples, begin + step, search)
        if cut <= begin:
            cut = begin + step
        bounds.append((begin, cut))
        begin = cut
    bounds.append((begin, total))
    return bounds


# ---- model ----------------------------------------------------------------------

def _is_memory_error(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or (
        isinstance(exc, RuntimeError) and bool(_MEMORY_ERROR_RE.search(str(exc)))
    )


def load_model(model: str, device: str = "cpu", compute_type: str = "int8",
               cpu_threads: Optional[int] = None, local_files_only: bool = False,
               status_callback: Optional[Callable[[str], None]] = None) -> Any:
    """Create a ``faster_whisper.WhisperModel`` (downloads Systran/faster-whisper-<model> on first use).

    Allocation failures (``mkl_malloc: failed to allocate memory``) are retried, since on a busy
    Windows machine they usually clear once the pagefile grows.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionSetupError(
            'faster-whisper is not installed. In the project venv run: pip install -e ".[transcribe]"'
        ) from exc
    for attempt in range(1, MEMORY_RETRIES + 1):
        try:
            return WhisperModel(model, device=device, compute_type=compute_type,
                                cpu_threads=cpu_threads or os.cpu_count() or 4,
                                local_files_only=local_files_only)
        except ValueError as exc:  # invalid model size / compute type
            raise TranscriptionError(f"Cannot load model '{model}': {exc}") from exc
        except Exception as exc:  # memory, network/proxy (huggingface_hub), missing cache, corrupt files
            if _is_memory_error(exc):
                if attempt < MEMORY_RETRIES:
                    if status_callback:
                        status_callback(f"Not enough memory to load '{model}' ({exc}); retry "
                                        f"{attempt}/{MEMORY_RETRIES - 1} in {MEMORY_RETRY_WAIT:.0f}s...")
                    time.sleep(MEMORY_RETRY_WAIT)
                    continue
                raise TranscriptionError(
                    f"Not enough memory to load model '{model}' ({exc}). Close applications, "
                    "use a smaller model or keep --compute-type int8."
                ) from exc
            source = model if Path(model).is_dir() else f"https://huggingface.co/Systran/faster-whisper-{model}"
            raise TranscriptionError(
                f"Cannot load model '{model}' from {source}: {type(exc).__name__}: {exc}. If the download "
                "failed, check network/proxy (HTTPS_PROXY, HF_ENDPOINT), drop --offline, or pass a local model dir."
            ) from exc
    raise AssertionError("unreachable")


# ---- outputs --------------------------------------------------------------------

def write_txt(segments: list[TranscriptSegment], path: Path) -> Path:
    lines = [f"[{format_timestamp(s.start)}] {s.text}" for s in segments]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return path


def write_srt(segments: list[TranscriptSegment], path: Path) -> Path:
    blocks = [
        f"{i}\n{format_timestamp(s.start, True, ',')} --> {format_timestamp(s.end, True, ',')}\n{s.text}\n"
        for i, s in enumerate(segments, 1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    return path


def _segment_to_json(seg: TranscriptSegment) -> dict[str, Any]:
    return {
        "start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text,
        "avg_logprob": round(seg.avg_logprob, 4), "no_speech_prob": round(seg.no_speech_prob, 4),
        "compression_ratio": round(seg.compression_ratio, 3), "low_confidence": seg.low_confidence,
    }


def load_transcript(json_path: str | Path) -> tuple[dict[str, Any], list[TranscriptSegment]]:
    """Read a ``<name>.json`` written by this module: (metadata, segments)."""
    path = Path(json_path).expanduser()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            TranscriptSegment(
                start=float(s["start"]), end=float(s["end"]), text=str(s["text"]),
                avg_logprob=float(s.get("avg_logprob", 0.0)), no_speech_prob=float(s.get("no_speech_prob", 0.0)),
                compression_ratio=float(s.get("compression_ratio", 0.0)),
            )
            for s in data.get("segments", [])
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TranscriptionError(f"Cannot read transcript JSON {path}: {exc}") from exc
    meta = {k: v for k, v in data.items() if k != "segments"}
    return meta, segments


def merge_segments(base: list[TranscriptSegment], overlay: list[TranscriptSegment],
                   start: float, end: float) -> list[TranscriptSegment]:
    """Replace ``base`` segments whose midpoint is inside ``[start, end)`` by ``overlay``.

    Typical use: re-transcribe a range with a bigger model and splice it into the full transcript.
    """
    kept = [s for s in base if not start <= (s.start + s.end) / 2 < end]
    return sorted(kept + list(overlay), key=lambda s: (s.start, s.end))


# ---- main entry point -------------------------------------------------------------

def transcribe_media(
    media_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = DEFAULT_LANGUAGE,
    start: str | float | None = None,
    end: str | float | None = None,
    beam_size: int = 5,
    initial_prompt: Optional[str] = None,
    carry_prompt: bool = False,
    vad_filter: bool = True,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: Optional[int] = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    output_name: Optional[str] = None,
    keep_audio: bool = False,
    local_files_only: bool = False,
    resume: bool = False,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> TranscriptionResult:
    """Transcribe the audio of ``media_path`` and write ``<name>.txt/.srt/.json`` into ``output_dir``.

    ``start``/``end`` (seconds or ``[hh:]mm:ss``) limit the window; timestamps stay absolute to the
    media. Decoding uses ``condition_on_previous_text=False`` (avoids hallucination loops), so
    ``initial_prompt`` only steers the first 30 s window of each chunk unless ``carry_prompt`` is set
    (prompt applied to every window via faster-whisper ``hotwords``). Outputs are rewritten after
    each chunk (``"complete": false`` until the end); ``resume`` continues an interrupted run.
    ``progress_callback(done_seconds, total_seconds)`` reports progress through the window.
    """
    t_total = time.perf_counter()

    def status(message: str) -> None:
        if status_callback:
            status_callback(message)

    # ---- 1. Validate arguments -------------------------------------------
    media = Path(media_path).expanduser().resolve()
    if not media.is_file():
        raise TranscriptionError(f"Media file not found: {media}")
    start_s = parse_time(start) or 0.0
    end_s = parse_time(end)
    if end_s is not None and end_s <= start_s:
        raise TranscriptionError(f"end ({end_s:.3f}s) must be greater than start ({start_s:.3f}s).")
    if beam_size < 1:
        raise TranscriptionError("beam_size must be >= 1.")
    if cpu_threads is not None and cpu_threads < 0:
        raise TranscriptionError("cpu_threads must be >= 0 (0 = all logical cores).")
    if chunk_seconds and 0 < chunk_seconds < 30:
        raise TranscriptionError("chunk_seconds must be 0 (no chunking) or >= 30.")
    language = None if language in (None, "", "auto") else language
    prompt = (initial_prompt or "").strip() or None
    name = output_name or media.stem

    out = Path(output_dir).expanduser()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TranscriptionError(f"Cannot create output directory '{out}': {exc}") from exc
    paths = {"txt": out / f"{name}.txt", "srt": out / f"{name}.srt", "json": out / f"{name}.json"}

    ffmpeg = find_ffmpeg()
    duration = probe_duration(media, ffmpeg)
    if duration is not None:
        if start_s >= duration:
            raise TranscriptionError(f"start ({start_s:.1f}s) is beyond the media duration ({duration:.1f}s).")
        end_s = duration if end_s is None else min(end_s, duration)

    # ---- 2. Resume an interrupted run -------------------------------------
    segments: list[TranscriptSegment] = []
    resumed_from: Optional[float] = None
    if resume and paths["json"].is_file():
        meta, previous = load_transcript(paths["json"])
        if meta.get("complete"):
            raise TranscriptionError(f"{paths['json']} is already complete; nothing to resume.")
        if meta.get("model") != model or Path(str(meta.get("media", ""))) != media:
            raise TranscriptionError(f"{paths['json']} was made for another media/model; cannot resume.")
        window = meta.get("window") or {}
        if start is None:
            start_s = float(window.get("start", start_s))
        if end is None and window.get("end") is not None:
            end_s = float(window["end"])
        resumed_from = float(meta.get("processed_until", start_s))
        segments = [s for s in previous if s.start < resumed_from]
        status(f"Resuming at {format_timestamp(resumed_from)} ({len(segments)} segments kept).")
    run_start = resumed_from if resumed_from is not None else start_s
    if end_s is not None and run_start >= end_s:
        raise TranscriptionError("Nothing left to transcribe in the requested window.")

    # ---- 3. Extract audio --------------------------------------------------
    wav = out / f"{name}.16k.wav"
    window_label = f"{format_timestamp(run_start)}-{format_timestamp(end_s) if end_s is not None else 'end'}"
    status(f"Extracting audio ({window_label}) with ffmpeg...")
    t0 = time.perf_counter()
    extract_audio(media, wav, run_start, end_s, ffmpeg)
    try:
        samples = read_wav(wav)
    finally:
        if not keep_audio:
            wav.unlink(missing_ok=True)
    t_extract = time.perf_counter() - t0
    audio_duration = len(samples) / SAMPLE_RATE
    if audio_duration < 0.1:
        raise TranscriptionError(f"No audio extracted from {media} in window {window_label}.")
    window_end = end_s if end_s is not None else run_start + audio_duration

    # ---- 4. Load the model -------------------------------------------------
    threads = cpu_threads or os.cpu_count() or 4
    status(f"Loading model '{model}' ({device}, {compute_type}, {threads} threads)...")
    t0 = time.perf_counter()
    whisper = load_model(model, device, compute_type, threads, local_files_only, status_callback)
    t_load = time.perf_counter() - t0

    # ---- 5. Transcribe chunk by chunk --------------------------------------
    bounds = plan_chunks(samples, chunk_seconds)
    status(f"Transcribing {audio_duration:.1f}s of audio in {len(bounds)} chunk(s)...")
    options: dict[str, Any] = {
        "language": language, "task": "transcribe", "beam_size": beam_size, "vad_filter": vad_filter,
        "condition_on_previous_text": False,
        "initial_prompt": None if carry_prompt else prompt,
        "hotwords": prompt if carry_prompt else None,
    }
    detected_language, language_probability, speech = language, 1.0, 0.0
    t0 = time.perf_counter()
    t_transcribe = 0.0
    chunk_list = [[round(run_start + a / SAMPLE_RATE, 3), round(run_start + b / SAMPLE_RATE, 3)] for a, b in bounds]

    def snapshot(processed_until: float, complete: bool) -> TranscriptionResult:
        result = TranscriptionResult(
            media=str(media), output_dir=str(out), files={k: str(v) for k, v in paths.items()},
            model=model, language=detected_language, language_probability=language_probability,
            duration=duration, window_start=start_s, window_end=window_end,
            audio_duration=processed_until - run_start, speech_duration=speech, segments=segments,
            timings={"extract": round(t_extract, 2), "load_model": round(t_load, 2),
                     "transcribe": round(t_transcribe, 2), "total": round(time.perf_counter() - t_total, 2)},
        )
        _write_outputs(result, paths, processed_until, complete, {
            "device": device, "compute_type": compute_type, "cpu_threads": threads, "beam_size": beam_size,
            "vad_filter": vad_filter, "condition_on_previous_text": False, "initial_prompt": prompt,
            "carry_prompt": carry_prompt, "chunk_seconds": chunk_seconds, "chunks": chunk_list,
            "resumed_from": resumed_from,
        })
        return result

    for a, b in bounds:
        offset = run_start + a / SAMPLE_RATE
        for attempt in range(1, MEMORY_RETRIES + 1):
            found: list[TranscriptSegment] = []
            try:
                chunk = samples[a:b].astype(np.float32) / 32768.0
                seg_iter, info = whisper.transcribe(chunk, **options)
                for seg in seg_iter:
                    text = seg.text.strip()
                    if text:
                        found.append(TranscriptSegment(
                            start=offset + seg.start, end=min(offset + seg.end, window_end), text=text,
                            avg_logprob=float(seg.avg_logprob), no_speech_prob=float(seg.no_speech_prob),
                            compression_ratio=float(seg.compression_ratio),
                        ))
                    if progress_callback:
                        progress_callback(min(a / SAMPLE_RATE + seg.end, audio_duration), audio_duration)
                break
            except Exception as exc:  # CTranslate2/MKL/onnxruntime failures surface as RuntimeError
                where = format_timestamp(offset)
                if _is_memory_error(exc) and attempt < MEMORY_RETRIES:
                    status(f"Out of memory in chunk at {where} ({exc}); retry {attempt}/{MEMORY_RETRIES - 1} "
                           f"in {MEMORY_RETRY_WAIT:.0f}s...")
                    time.sleep(MEMORY_RETRY_WAIT)
                    continue
                raise TranscriptionError(
                    f"Transcription failed in chunk at {where}: {type(exc).__name__}: {exc}. "
                    "Earlier chunks are saved; rerun with --resume."
                ) from exc
            finally:
                chunk = None
        segments.extend(found)
        detected_language, language_probability = info.language, float(info.language_probability)
        speech += float(getattr(info, "duration_after_vad", 0.0) or 0.0)
        t_transcribe = time.perf_counter() - t0
        if progress_callback:
            progress_callback(b / SAMPLE_RATE, audio_duration)
        result = snapshot(run_start + b / SAMPLE_RATE, complete=(b == len(samples)))
    return result


def _write_outputs(result: TranscriptionResult, paths: dict[str, Path], processed_until: float,
                   complete: bool, settings: dict[str, Any]) -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        fw_version = version("faster-whisper")
    except PackageNotFoundError:
        fw_version = None
    write_txt(result.segments, paths["txt"])
    write_srt(result.segments, paths["srt"])
    payload = {
        "tool": "frame_extractor.transcribe", "complete": complete, "media": result.media,
        "duration": result.duration, "window": {"start": result.window_start, "end": result.window_end},
        "processed_until": round(processed_until, 3), "model": result.model,
        "language": result.language, "language_probability": round(result.language_probability, 4),
        **settings,
        "audio_duration": round(result.audio_duration, 3), "speech_duration": round(result.speech_duration, 3),
        "elapsed": result.timings.get("total"), "timings": result.timings,
        "throughput": round(result.throughput, 3), "faster_whisper_version": fw_version,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "segments": [_segment_to_json(s) for s in result.segments],
    }
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- CLI --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-transcribe",
        description="Transcribe the audio of a video/audio file with faster-whisper (txt, srt, json).",
    )
    p.add_argument("media", help="Input video/audio file (anything ffmpeg can read)")
    p.add_argument("-o", "--output-dir",
                   help="Output directory (default: <Documents>/frame-extractor/transcripts)")
    p.add_argument("--name", help="Output file name without extension (default: media file name)")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL,
                   help="Whisper size (tiny, base, small, medium, large-v3, turbo...) or local model dir "
                        f"(default: {DEFAULT_MODEL})")
    p.add_argument("-l", "--language", default=DEFAULT_LANGUAGE,
                   help=f"Spoken language code, or 'auto' to detect (default: {DEFAULT_LANGUAGE})")
    p.add_argument("--start", help="Window start: seconds or [hh:]mm:ss (default: beginning)")
    p.add_argument("--end", help="Window end: seconds or [hh:]mm:ss (default: end of media)")
    p.add_argument("-b", "--beam-size", type=int, default=5, help="Beam size (default: 5)")
    prompt = p.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="Initial prompt with domain vocabulary / spelling hints")
    prompt.add_argument("--prompt-file", help="UTF-8 text file with the initial prompt")
    p.add_argument("--carry-prompt", action="store_true",
                   help="Apply the prompt to every 30 s window (faster-whisper hotwords), not just the first")
    p.add_argument("--no-vad", action="store_true", help="Disable the Silero VAD filter (not recommended)")
    p.add_argument("--device", default="cpu", help="cpu, cuda or auto (default: cpu)")
    p.add_argument("--compute-type", default="int8", help="CTranslate2 compute type (default: int8)")
    p.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 4,
                   help=f"CPU threads (default: logical cores = {os.cpu_count()})")
    p.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
                   help=f"Process audio in chunks cut at pauses; 0 = single pass (default: {DEFAULT_CHUNK_SECONDS:.0f})")
    p.add_argument("--keep-audio", action="store_true", help="Keep the extracted <name>.16k.wav")
    p.add_argument("--offline", action="store_true", help="Use only an already-downloaded model")
    p.add_argument("--resume", action="store_true", help="Continue an interrupted run from <name>.json")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # cp1252 consoles/pipes must not crash on any text
    # hf-xet (Rust) aborted the process with "memory allocation failed" under memory pressure;
    # plain HTTPS downloads are robust. An explicit HF_HUB_DISABLE_XET=0 still wins.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = build_parser().parse_args(argv)

    prompt = args.prompt
    if args.prompt_file:
        try:
            prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8-sig").strip()
        except OSError as exc:
            print(f"Error: cannot read prompt file: {exc}", file=sys.stderr)
            return 2

    from tqdm import tqdm

    bar: Optional[tqdm] = None

    def on_progress(current: float, total: float) -> None:
        nonlocal bar
        if bar is None:
            bar = tqdm(total=round(total, 1), unit="s", desc="Transcribing", dynamic_ncols=True,
                       bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} s audio [{elapsed}<{remaining}]")
        delta = round(current, 1) - bar.n
        if delta > 0:
            bar.update(delta)

    def on_status(message: str) -> None:
        tqdm.write(message, file=sys.stderr)

    try:
        result = transcribe_media(
            args.media, args.output_dir or transcripts_root(), model=args.model, language=args.language,
            start=args.start, end=args.end, beam_size=args.beam_size, initial_prompt=prompt,
            carry_prompt=args.carry_prompt, vad_filter=not args.no_vad, device=args.device,
            compute_type=args.compute_type, cpu_threads=args.threads, chunk_seconds=args.chunk_seconds,
            output_name=args.name, keep_audio=args.keep_audio, local_files_only=args.offline,
            resume=args.resume, progress_callback=on_progress, status_callback=on_status,
        )
    except TranscriptionSetupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Outputs written so far are marked \"complete\": false; rerun with --resume.",
              file=sys.stderr)
        return 130
    finally:
        if bar is not None:
            bar.close()

    low = sum(1 for s in result.segments if s.low_confidence)
    print(f"Transcribed {result.audio_duration:.1f}s of audio ({result.speech_duration:.1f}s speech) "
          f"in {result.timings['transcribe']:.1f}s = {result.throughput:.2f}x real time "
          f"[model {result.model}, language {result.language}]: {len(result.segments)} segments, "
          f"{low} low-confidence")
    for kind, path in result.files.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
