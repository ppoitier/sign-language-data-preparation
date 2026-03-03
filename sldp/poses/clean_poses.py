from typing import Optional, Callable
import numpy as np
import tarfile
from tqdm import tqdm

from sign_language_tools.pose.mediapipe.vertices import (
    LIPS_VERTICES,
    LEFT_EYE_VERTICES,
    RIGHT_EYE_VERTICES,
    LEFT_IRIS_VERTICES,
    RIGHT_IRIS_VERTICES,
    LEFT_EYEBROW_VERTICES,
    RIGHT_EYEBROW_VERTICES,
)

from sign_language_tools.common.transforms import Compose, ReplaceNaN
from sign_language_tools.pose.transform import InterpolateMissing

from sldp.poses.io import iter_poses_from_tars, add_poses_to_tar


def only_keep_relevant_body_parts(
    pose_sequences: dict[str, np.ndarray],
    body_parts: Optional[set[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Filters MediaPipe landmarks to retain only specified body parts.

    This function extracts specific sub-sets of landmarks (like lips or eyes) from the
    raw extracted holistic landmarks and maps them to top-level keys in the output dictionary.

    Args:
        pose_sequences: A dictionary containing the raw pose sequences.
            Expected keys are 'pose', 'left_hand', 'right_hand', 'face'.
            Values are ndarrays of shape (T, L, C) where:
            - T: Number of frames
            - L: Number of landmarks
            - C: Number of coordinates (x, y, z, etc.)

        body_parts: A set of strings defining which parts to keep.
            Options include: 'upper_pose', 'left_hand', 'right_hand', 'lips',
            'left_eye', 'right_eye', 'left_iris', 'right_iris',
            'left_eyebrow', 'right_eyebrow'.

            If None, defaults to all available parts.

    Returns:
        Dict[str, np.ndarray]: A new dictionary containing only the requested body parts.

    Example:
        >>> seq = {'face': np.zeros((10, 468, 3)), ...}
        >>> res = only_keep_relevant_body_parts(seq, body_parts={'lips'})
        >>> res['lips'].shape
        (10, 40, 3)
    """
    definition_map = {
        "left_hand": ("left_hand", slice(None)),  # Keep all
        "right_hand": ("right_hand", slice(None)),  # Keep all
        "upper_pose": ("pose", slice(0, 23)),
        "lips": ("face", LIPS_VERTICES),
        "left_eye": ("face", LEFT_EYE_VERTICES),
        "right_eye": ("face", RIGHT_EYE_VERTICES),
        "left_iris": ("face", LEFT_IRIS_VERTICES),
        "right_iris": ("face", RIGHT_IRIS_VERTICES),
        "left_eyebrow": ("face", LEFT_EYEBROW_VERTICES),
        "right_eyebrow": ("face", RIGHT_EYEBROW_VERTICES),
    }
    if body_parts is None:
        body_parts = set(definition_map.keys())

    new_pose_sequence = {}
    for part in body_parts:
        if part not in definition_map:
            continue
        source_key, indices = definition_map[part]
        if source_key in pose_sequences:
            # Note: We use the colon : in the first dimension to keep all Frames (T)
            new_pose_sequence[part] = pose_sequences[source_key][:, indices]

    return new_pose_sequence


def fill_missing_landmarks(
    pose_sequences: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    transform = Compose(
        [
            InterpolateMissing(method="linear"),
            ReplaceNaN(fill_value=0.0),
        ]
    )
    return {k: transform(v).astype("float16") for k, v in pose_sequences.items()}


def clean_poses(pose_sequences: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return fill_missing_landmarks(only_keep_relevant_body_parts(pose_sequences))


def clean_all_poses_from_tars(
    source_tar_urls: str,
    dest_tar_path: str,
    filter_func: Optional[Callable] = None,
    show_progress=False,
):
    tar = tarfile.open(dest_tar_path, mode="w|")
    for sample_id, poses in tqdm(
        iter_poses_from_tars(source_tar_urls), disable=not show_progress
    ):
        if filter_func is not None and not filter_func(sample_id, poses):
            continue
        add_poses_to_tar(sample_id, clean_poses(poses), tar)
    tar.close()
