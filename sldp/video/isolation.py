import os
import subprocess
import tempfile

from tqdm import tqdm

from sldp.utils.json import from_json
from sldp.utils.tar import create_inmemory_tar, save_inmemory_tar, add_file_to_tar, load_bytes_from_tar


def _cut_single_clip(
    video_path: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    use_gpu: bool = True,
) -> bytes:
    """
    Cut a single segment from a video file and return it as bytes.

    The end_frame is inclusive, so (1000, 1050) produces 51 frames.
    Output is a fragmented MP4 written to stdout (pipe-friendly).
    """
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]

    if use_gpu:
        cmd += [
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-c:v", "h264_cuvid",
        ]

    # -ss before -i enables fast seeking
    cmd += [
        "-ss", f"{start_time:.6f}",
        "-i", video_path,
        "-t", f"{duration:.6f}",
        "-an",
    ]

    if use_gpu:
        cmd += ["-c:v", "h264_nvenc"]
    else:
        cmd += ["-c:v", "libx264"]

    # frag_keyframe+empty_moov is required to write MP4 to a pipe
    # (the regular MP4 muxer needs a seekable output for the moov atom)
    cmd += ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on frames [{start_frame}, {end_frame}]: "
            f"{result.stderr.decode()}"
        )
    return result.stdout


def _cut_clips(
    video_path: str,
    segments: list[tuple[int, int]],
    fps: float,
    use_gpu: bool = True,
    progress: bool = False,
) -> list[bytes]:
    """
    Cut multiple isolated signs from a video.

    Args:
        video_path: Source video file path.
        segments:    List of (start_frame, end_frame) tuples.
                     end_frame is inclusive.
        use_gpu:     Use NVIDIA h264_cuvid / h264_nvenc acceleration.
        progress:   Whether to show a progress bar. Defaults to False.

    Returns:
        A list of bytes, one per segment, each a valid MP4.
    """
    results = []
    for start_frame, end_frame in tqdm(segments, unit="seg", disable=not progress):
        segment = _cut_single_clip(
            video_path, start_frame, end_frame, fps, use_gpu
        )
        results.append(segment)
    return results


def create_clips_from_video_tar(
        segments: dict[str, list[tuple[int, int]]],
        video_dir: str,
        dest_video_tar_path: str,
        fps: float,
):
    tar, tar_buffer = create_inmemory_tar()
    for entry_name, segments in segments.items():
        print('Extracting clips:', entry_name)
        clips = _cut_clips(f"{video_dir}/{entry_name}.mp4", segments, fps=fps, use_gpu=True, progress=True)
        for (start_frame, end_frame), clip_bytes in zip(segments, clips):
            clip_filename = f"{entry_name}_{start_frame}_{end_frame}.mp4"
            add_file_to_tar(clip_filename, tar, clip_bytes)
        break
    save_inmemory_tar(dest_video_tar_path, tar, tar_buffer)
