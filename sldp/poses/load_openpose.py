import dataclasses
import tarfile
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class Pose:
    id: str
    n_frames: int
    n_coords: int
    body_regions: tuple[str, ...]
    poses: dict[str, np.ndarray]
    frame_statuses: list[str]


def _get_pose_from_signer_data(signer_data: dict, body_region: str) -> np.ndarray:
    match body_region:
        case "pose" | "face":
            key = body_region
        case "left_hand":
            key = "hand_left"
        case "right_hand":
            key = "hand_right"
        case _:
            raise ValueError(f"Unknown body region: [{body_region}].")
    key = f"{key}_keypoints_2d"
    return np.array(signer_data[key], dtype='float16').reshape(-1, 3)


def _get_empty_pose(body_regions: tuple[str, ...], n_coords: int):
    poses = {}
    for region in body_regions:
        match region:
            case "pose":
                n_landmarks = 25
            case "left_hand" | "right_hand":
                n_landmarks = 21
            case "face":
                n_landmarks = 70
            case _:
                raise ValueError(f"Unknown body region: [{region}].")
        poses[region] = np.full(
            (n_landmarks, n_coords), fill_value=np.nan, dtype="float16"
        )
    return poses


def _merge_poses(poses) -> tuple[dict[str, np.ndarray], list[str]]:
    indices, frame_poses = zip(*sorted(poses.items(), key=lambda x: x[0]))
    body_regions = frame_poses[0][0].keys()
    merged_poses = {
        region: np.stack([pose[region] for pose, _ in frame_poses], axis=0)
        for region in body_regions
    }
    merged_status = [status for _, status in frame_poses]
    return merged_poses, merged_status


def read_open_pose_frame(
    frame_data: dict, body_regions=("pose", "left_hand", "right_hand"), n_coords=3
) -> tuple[dict[str, np.ndarray], str]:
    status = "ok"
    if len(frame_data["people"]) < 1:
        status = "missing-person"
    elif len(frame_data["people"]) > 1:
        status = "multiple-people"
    if status != "ok":
        return _get_empty_pose(body_regions, n_coords), status
    signer_data = frame_data["people"][0]
    poses = {region: _get_pose_from_signer_data(signer_data, region) for region in body_regions}
    return poses, status


def read_open_pose_tar(
    tar_filepath: str,
    show_progress=False,
    body_regions=("pose", "left_hand", "right_hand"),
    n_coords=3,
):
    tar_filepath = Path(tar_filepath)
    gzip = tar_filepath.name.endswith(".tar.gz")
    current_sample_id = None
    current_poses = dict()
    with tarfile.open(tar_filepath, "r|gz" if gzip else "r|") as tar:
        for member in tqdm(
            tar,
            desc=f"Reading OpenPose files [{tar_filepath.name}]",
            unit=" files",
            disable=not show_progress,
        ):
            if member.isfile() and member.name.endswith("_keypoints.json"):
                extracted_file = tar.extractfile(member)
                if extracted_file is not None:
                    frame_poses = read_open_pose_frame(
                        orjson.loads(extracted_file.read()),
                        body_regions=body_regions,
                        n_coords=n_coords,
                    )
                    sample_id, frame_nb, _ = member.name.split("/")[-1].rsplit("_", 2)

                    if current_sample_id is None:
                        current_sample_id = sample_id

                    if current_sample_id != sample_id:
                        merged_poses, merged_status = _merge_poses(current_poses)
                        yield Pose(
                            id=current_sample_id,
                            n_frames=len(current_poses),
                            n_coords=n_coords,
                            body_regions=body_regions,
                            poses=merged_poses,
                            frame_statuses=merged_status,
                        )
                        current_sample_id = sample_id
                        current_poses = dict()

                    current_poses[int(frame_nb)] = frame_poses
                else:
                    raise ValueError(f"Could not extract file [{member.name}].")
    if current_sample_id is not None and current_poses:
        merged_poses, merged_status = _merge_poses(current_poses)
        yield Pose(
            id=current_sample_id,
            n_frames=len(current_poses),
            n_coords=n_coords,
            body_regions=body_regions,
            poses=merged_poses,
            frame_statuses=merged_status,
        )
