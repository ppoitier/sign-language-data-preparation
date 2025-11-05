from pathlib import Path
from typing import TypedDict

import numpy as np
import sign_language_tools.pose.mediapipe.extraction as mp_extractor

from sldp.utils.parallel import run_parallel


class PoseExtractionCommand(TypedDict):
    sample_id: str
    src_video_path: str
    dest_poses_dir: str


def build_poses_from_sample(
        sample_id: str,
        src_video_path: str,
        dest_poses_dir: str,
):
    dest_poses_dir = Path(dest_poses_dir)
    src_video_ext = Path(src_video_path).suffix
    poses = mp_extractor.extract_poses_from_video(src_video_path, show_progress=True)
    for region, region_poses in poses.items():
        pose_path = dest_poses_dir / region / f"{sample_id}.{src_video_ext}"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pose_path, region_poses)


def build_poses_from_samples(commands: list[PoseExtractionCommand], n_jobs: int = 8):
    commands: list[dict]
    run_parallel(build_poses_from_sample, commands, n_jobs=n_jobs)
