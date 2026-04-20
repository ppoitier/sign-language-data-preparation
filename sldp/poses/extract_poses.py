import os
from pathlib import Path
from typing import Sequence, Optional, Literal
from contextlib import ExitStack
from collections import ChainMap
from itertools import batched

from pydantic import BaseModel
import pandas as pd

from sldp.utils.tar import create_inmemory_tar, add_file_to_tar, save_inmemory_tar
from sldp.utils.parallel import run_parallel
from sldp.poses.mediapipe.extraction import load_holistic_landmarker, extract_poses

__all__ = ["batch_extract_all_poses_from_video_dir"]


class PoseExtractionCommand(BaseModel):
    sample_id: str
    src_video_path: str
    dest_poses_dir: str


class BatchPoseExtractionCommand(BaseModel):
    sample_ids: Sequence[str]
    src_video_paths: Sequence[str]
    dest_tar_path: str
    landmarker_path: str


def _process_batch_pose_extraction_command(
    command: BatchPoseExtractionCommand,
    show_progress: bool = False,
    verbose: bool = False,
) -> dict[str, tuple[str, Optional[str]]]:
    dest_tar, dest_tar_buffer = create_inmemory_tar()
    extraction_statuses: dict[str, tuple[str, Optional[str]]] = dict()
    for filepath, sample_id in zip(command.src_video_paths, command.sample_ids):
        try:
            with ExitStack() as stack:
                landmarker = load_holistic_landmarker(command.landmarker_path)
                # Register landmarker so they close when the block exits
                if hasattr(landmarker, "__enter__"):
                    stack.enter_context(landmarker)
                poses = extract_poses(
                    filepath,
                    landmarker,
                    show_progress=show_progress,
                )
                for body_part, poses_array in poses.items():
                    add_file_to_tar(
                        f"{sample_id}.poses.{body_part}.npy", dest_tar, poses_array
                    )
                extraction_statuses[sample_id] = "ok", None
                if verbose:
                    print(f"[{os.getpid()}] Success: {sample_id}")
        except Exception as e:
            extraction_statuses[sample_id] = "error", str(e)
            if verbose:
                print(f"[{os.getpid()}] FAILED: {sample_id} | Error: {e}")
    save_inmemory_tar(command.dest_tar_path, dest_tar, dest_tar_buffer)
    return extraction_statuses


def batch_extract_all_poses_from_video_dir(
    video_dir: str,
    dest_poses_dir: str,
    landmarker_path: str,
    tar_name="poses_{:0>6}.tar",
    n_workers: int = 4,
    max_poses_per_tar: int = 500,
    accepted_extensions: tuple[str, ...] = ("mp4", "avi", "webm"),
    index_offset=0,
    show_progress: bool = False,
    verbose: bool = False,
    samples_to_skip: Optional[set[str]] = None,
):
    video_paths = [
        entry.path
        for entry in os.scandir(video_dir)
        if entry.is_file() and (entry.name.split(".")[-1] in accepted_extensions)
    ]
    n_total_videos = len(video_paths)
    n_skipped_videos = 0
    if samples_to_skip is not None:
        video_paths = [p for p in video_paths if Path(p).stem not in samples_to_skip]
        n_skipped_videos = n_total_videos - len(video_paths)
    video_path_batches = list(
        batched(video_paths, n=min(len(video_paths) // n_workers, max_poses_per_tar))
    )
    if verbose:
        print(f"Skipped {n_skipped_videos} samples.")
        print(
            f"Prepare to extract poses from {len(video_paths)} videos using {len(video_path_batches)} batches."
        )
    job_kwargs = [
        dict(
            command=BatchPoseExtractionCommand(
                src_video_paths=batch_video_paths,
                sample_ids=[Path(v).stem for v in batch_video_paths],
                dest_tar_path=dest_poses_dir + "/" + tar_name.format(i + index_offset),
                landmarker_path=landmarker_path,
            ),
            show_progress=show_progress,
            verbose=verbose,
        )
        for i, batch_video_paths in enumerate(video_path_batches)
    ]
    extraction_statuses = run_parallel(
        _process_batch_pose_extraction_command, kwargs_list=job_kwargs, n_jobs=n_workers
    )
    extraction_statuses = dict(ChainMap(*extraction_statuses))
    df = pd.DataFrame(
        [
            {"id": sample_id, "status": status, "error_msg": msg}
            for sample_id, (status, msg) in extraction_statuses.items()
        ]
    )
    return df
