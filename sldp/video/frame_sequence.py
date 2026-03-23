"""
Utilities for working with image-based frame sequences as video.

This module provides tools to group flat file listings (e.g. a tar index)
into per-video frame sequences and to encode those sequences into MP4
files via ffmpeg — all without extracting frames to disk.

Both CPU (``libx264``) and NVIDIA GPU (``h264_nvenc``) encoders are
supported.

Typical workflow::

    from mylib.tar import load_index
    from mylib.video.frame_sequence import (
        group_index_as_frame_sequences,
        encode_frames_to_video,
    )

    index = load_index("dataset.json")
    sequences = group_index_as_frame_sequences(index)

    for folder, frames in sequences.items():
        encode_frames_to_video("dataset.tar", frames, f"videos/{folder}.mp4")
"""

import os
import re
import subprocess
from collections import defaultdict
from typing import Callable, Literal

from tqdm import tqdm


FrameKey = Callable[[str], int]
"""A callable that extracts an integer sort key from a frame filename."""


def frame_key_from_pattern(pattern: str) -> FrameKey:
    """Build a :data:`FrameKey` from a regex with one capturing group.

    The first capturing group must match a decimal integer (the frame
    number).

    Args:
        pattern: A regex string with exactly one capturing group.

    Returns:
        A callable ``(filename) -> int`` suitable for the *frame_key*
        parameter of :func:`group_index_as_frame_sequences`.

    Example::

        # Phoenix-2014 style:  …_fn000117-0.png  →  117
        key = frame_key_from_pattern(r"fn(\\d+)")

        # Simple sequential:   frame_00042.png  →  42
        key = frame_key_from_pattern(r"frame_(\\d+)")
    """
    compiled = re.compile(pattern)

    def _extract(filename: str) -> int:
        m = compiled.search(filename)
        if m is None:
            raise ValueError(
                f"Cannot extract frame number from {filename!r} "
                f"using pattern {pattern!r}"
            )
        return int(m.group(1))

    return _extract


#: Default frame key for the Phoenix-2014 dataset (``…_fn000117-0.png``).
PHOENIX_FRAME_KEY: FrameKey = frame_key_from_pattern(r"fn(\d+)")


def group_index_as_frame_sequences(
    index: dict[str, tuple[int, int]],
    frame_key: FrameKey = PHOENIX_FRAME_KEY,
    extension: str = ".png",
    progress: bool = False,
) -> dict[str, list[tuple[str, int, int]]]:
    """Group a tar index into per-video frame sequences sorted by frame number.

    Takes a flat tar index (as produced by ``get_tar_index``) and groups
    entries whose parent directory is the same — each such directory is
    assumed to hold the frames of a single video.  Within each group the
    frames are sorted by the integer key returned by *frame_key*.

    Args:
        index:
            Tar index mapping member names to ``(byte_offset, byte_size)``.
        frame_key:
            A callable ``(filename) -> int`` that returns the sort key for
            a given frame filename.  Defaults to :data:`PHOENIX_FRAME_KEY`.
            Use :func:`frame_key_from_pattern` to build one from a regex,
            or pass any custom callable.
        extension:
            Only members whose name ends with this suffix
            (case-insensitive) are considered.  Defaults to ``".png"``.
        progress:
            If *True*, display a ``tqdm`` progress bar while iterating
            the index.

    Returns:
        A dictionary mapping each video folder path to an ordered list of
        ``(member_name, byte_offset, byte_size)`` tuples, sorted by
        ascending frame number.

    Example::

        # Phoenix-2014 (default key)
        sequences = group_index_as_frame_sequences(index)

        # Custom dataset with filenames like frame_00042.png
        key = frame_key_from_pattern(r"frame_(\\d+)")
        sequences = group_index_as_frame_sequences(index, frame_key=key)

        # Fully custom logic
        sequences = group_index_as_frame_sequences(
            index,
            frame_key=lambda f: int(f.split("_")[2]),
        )
    """
    groups: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)

    items = index.items()
    if progress:
        items = tqdm(items, desc="Grouping frames", unit=" entries")

    for member_name, (offset, size) in items:
        if not member_name.lower().endswith(extension):
            continue
        folder = os.path.dirname(member_name)
        sort_key = frame_key(os.path.basename(member_name))
        groups[folder].append((member_name, offset, size, sort_key))

    # Sort each group by frame number and strip the sort key.
    sorted_groups: dict[str, list[tuple[str, int, int]]] = {}
    for folder, entries in groups.items():
        entries.sort(key=lambda x: x[3])
        sorted_groups[folder] = [(name, off, sz) for name, off, sz, _ in entries]

    return sorted_groups


CpuPreset = Literal[
    "ultrafast", "superfast", "veryfast", "faster",
    "fast", "medium", "slow", "slower", "veryslow",
]
NvencPreset = Literal["p1", "p2", "p3", "p4", "p5", "p6", "p7"]


def _build_ffmpeg_cmd(
    output_path: str,
    fps: int,
    gpu: bool,
    quality: int,
    preset: CpuPreset | NvencPreset | None,
    scale_height: int | None,
    keyframe_interval: int | None,
    pix_fmt: str,
    loglevel: str,
) -> list[str]:
    """Build the ffmpeg command list (internal helper)."""
    cmd: list[str] = [
        "ffmpeg", "-y",
        "-f", "image2pipe",
        "-framerate", str(fps),
        "-i", "pipe:0",
    ]

    # ---- Video filter (scaling) -------------------------------------------
    if scale_height is not None:
        cmd += ["-vf", f"scale=-2:{scale_height}"]

    # ---- Codec & quality --------------------------------------------------
    if gpu:
        effective_preset = preset or "p4"
        cmd += [
            "-c:v", "h264_nvenc",
            "-preset", str(effective_preset),
            "-rc", "vbr",
            "-cq", str(quality),
            "-b:v", "0",
        ]
    else:
        effective_preset = preset or "medium"
        cmd += [
            "-c:v", "libx264",
            "-preset", str(effective_preset),
            "-crf", str(quality),
        ]

    if keyframe_interval is not None:
        cmd += ["-g", str(keyframe_interval)]

    # ---- Common output options --------------------------------------------
    cmd += [
        "-pix_fmt", pix_fmt,
        "-movflags", "+faststart",
        "-an",
        "-loglevel", loglevel,
        output_path,
    ]
    return cmd


def encode_frames_to_video(
    tar_path: str,
    frames: list[tuple[str, int, int]],
    output_path: str,
    fps: int = 25,
    gpu: bool = False,
    quality: int | None = None,
    preset: CpuPreset | NvencPreset | None = None,
    scale_height: int | None = None,
    keyframe_interval: int | None = None,
    pix_fmt: str = "yuv420p",
    loglevel: str = "error",
) -> None:
    """Pipe image frames from a tar file into ffmpeg to produce an MP4.

    Frames are read as raw bytes from *tar_path* using each entry's byte
    offset and size, then streamed sequentially into ffmpeg's ``stdin``
    via the ``image2pipe`` demuxer.  No temporary files are written.

    Two encoder paths are supported:

    * **CPU** (``gpu=False``, default) — uses ``libx264`` with CRF-based
      quality control.  Presets range from ``"ultrafast"`` to
      ``"veryslow"`` (default ``"medium"``).
    * **GPU** (``gpu=True``) — uses NVIDIA's ``h264_nvenc`` with VBR
      rate control and a constant-quality (CQ) target.  Presets range
      from ``"p1"`` (fastest) to ``"p7"`` (best quality, default
      ``"p4"``).  Requires an NVIDIA GPU with NVENC support and the
      appropriate ffmpeg build.

    Args:
        tar_path:
            Path to the tar file containing the image frames.
        frames:
            Ordered list of ``(member_name, byte_offset, byte_size)``
            tuples, one per frame, in the desired playback order.
        output_path:
            Destination path for the encoded ``.mp4`` file.  Parent
            directories are created automatically.
        fps:
            Frames per second.  Defaults to ``25`` (PAL).
        gpu:
            If *True*, encode with ``h264_nvenc`` (NVIDIA GPU).
            Defaults to *False* (``libx264`` on CPU).
        quality:
            Encoder quality target.  For CPU this is the CRF value
            (0–51, lower is better, default ``23``).  For GPU this is
            the CQ value (0–51, lower is better, default ``32``).
        preset:
            Encoder preset.  When *None* a sensible default is chosen
            (``"medium"`` for CPU, ``"p4"`` for GPU).
        scale_height:
            If given, scale the video to this height in pixels while
            preserving the aspect ratio (width is auto-calculated to the
            nearest even value).  Useful for normalising frame sizes for
            training pipelines.  *None* means no scaling.
        keyframe_interval:
            Maximum number of frames between keyframes (I-frames).  A
            shorter interval allows data loaders to seek into the video
            more efficiently, at a small cost in file size.  For example,
            ``150`` at 50 fps places a keyframe every 3 seconds.  *None*
            (default) uses the encoder's built-in default.
        pix_fmt:
            Output pixel format.  ``"yuv420p"`` (default) ensures the
            widest player and decoder compatibility.
        loglevel:
            ffmpeg log verbosity (``"error"``, ``"warning"``, ``"info"``,
            …).  Defaults to ``"error"``.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.

    Example::

        encode_frames_to_video(
            tar_path="phoenix.tar",
            frames=sequences["some/video/folder"],
            output_path="out/video.mp4",
            fps=25,
            gpu=True,
            quality=30,
            scale_height=480,
        )
    """
    if quality is None:
        quality = 32 if gpu else 23

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = _build_ffmpeg_cmd(
        output_path=output_path,
        fps=fps,
        gpu=gpu,
        quality=quality,
        preset=preset,
        scale_height=scale_height,
        keyframe_interval=keyframe_interval,
        pix_fmt=pix_fmt,
        loglevel=loglevel,
    )

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    with open(tar_path, "rb") as tar_fh:
        for _name, offset, size in frames:
            tar_fh.seek(offset)
            proc.stdin.write(tar_fh.read(size))

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(
            f"ffmpeg exited with code {ret} while encoding {output_path}"
        )
