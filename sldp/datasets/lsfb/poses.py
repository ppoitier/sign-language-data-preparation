import os
from itertools import batched
from typing import Optional
from collections import ChainMap
from pathlib import Path
from contextlib import ExitStack
import pandas as pd

from sldp.utils.parallel import run_parallel
from sldp.utils.tar import create_inmemory_tar, save_inmemory_tar, add_file_to_tar
from sldp.poses.mediapipe.extraction import load_landmarkers, extract_poses


def _poses_extraction_job(
    video_filepaths: list[str],
    sample_ids: list[str],
    dest_tar_path: str,
    landmarker_paths: dict[str, str],
    show_progress: bool = False,
    verbose: bool = False,
) -> dict[str, tuple[str, Optional[str]]]:
    assert ("pose" in landmarker_paths) and (
        "hand" in landmarker_paths
    ), "Pose and hand landmarkers are mandatory."
    dest_tar, dest_tar_buffer = create_inmemory_tar()
    extraction_statuses: dict[str, tuple[str, Optional[str]]] = dict()
    for filepath, sample_id in zip(video_filepaths, sample_ids):
        try:
            with ExitStack() as stack:
                landmarkers = load_landmarkers(landmarker_paths)
                # Register models so they close when the block exits
                for model in landmarkers.values():
                    if hasattr(model, "__enter__"): # Safety check
                        stack.enter_context(model)
                poses = extract_poses(
                    filepath,
                    landmarkers["pose"],
                    landmarkers["hand"],
                    landmarkers.get("face"),
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
    save_inmemory_tar(dest_tar_path, dest_tar, dest_tar_buffer)
    return extraction_statuses


def extract_all_poses(
    video_dir: str,
    dest_dir: str,
    landmarker_paths: dict[str, str],
    tar_name="poses_{:0>6}.tar",
    n_workers: int = 4,
    max_poses_per_tar: int = 500,
    accepted_extensions: tuple[str, ...] = ("mp4", "avi", "webm"),
    show_progress: bool = False,
    verbose: bool = False,
):
    video_paths = [
        entry.path
        for entry in os.scandir(video_dir)
        if entry.is_file() and entry.name.split(".")[-1] in accepted_extensions
    ]
    video_path_batches = list(batched(video_paths, n=min(len(video_paths) // n_workers, max_poses_per_tar)))
    job_kwargs = [
        dict(
            video_filepaths=batch_video_paths,
            sample_ids=[Path(v).stem for v in batch_video_paths],
            dest_tar_path=dest_dir + "/" + tar_name.format(i),
            landmarker_paths=landmarker_paths,
            show_progress=show_progress,
            verbose=verbose,
        )
        for i, batch_video_paths in enumerate(video_path_batches)
    ]
    extraction_statuses = run_parallel(
        _poses_extraction_job, kwargs_list=job_kwargs, n_jobs=n_workers
    )
    extraction_statuses = dict(ChainMap(*extraction_statuses))
    df = pd.DataFrame([
        {"id": sample_id, "status": status, "error_msg": msg}
        for sample_id, (status, msg) in extraction_statuses.items()
    ])
    return df


if __name__ == "__main__":
    extract_all_poses(
        # video_dir="E:/datasets/sign-language/lsfb-cont/videos",
        video_dir="E:/datasets/sign-language/tmp/videos",
        dest_dir="E:/datasets/sign-language/tmp/poses",
        landmarker_paths={
            "hand": "C:/mediapipe/models/hand_landmarker.task",
            "pose": "C:/mediapipe/models/pose_landmarker_full.task",
            "face": "C:/mediapipe/models/face_landmarker.task",
        },
        verbose=True,
    )
