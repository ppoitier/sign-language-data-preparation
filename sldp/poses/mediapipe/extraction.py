from collections import defaultdict

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
)

from sldp.video.frame_iteration import iterate_video_frames_using_vidgear


def load_hand_landmarker(model_path: str):
    base_options = BaseOptions(
        model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU
    )
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    return HandLandmarker.create_from_options(options)


def load_pose_landmarker(model_path: str):
    base_options = BaseOptions(
        model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU
    )
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.2,
        min_tracking_confidence=0.2,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)


def load_face_landmarker(model_path: str):
    base_options = BaseOptions(
        model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU
    )
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.2,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def load_landmarkers(model_paths: dict[str, str]) -> dict[str, ...]:
    landmarkers = {}
    for body_part, model_path in model_paths.items():
        match body_part:
            case "hand":
                landmarkers[body_part] = load_hand_landmarker(model_path)
            case "pose":
                landmarkers[body_part] = load_pose_landmarker(model_path)
            case "face":
                landmarkers[body_part] = load_face_landmarker(model_path)
            case _:
                raise ValueError(f"Unknown landmarker: {body_part}")
    return landmarkers


def _landmarks_to_numpy(landmarks) -> np.ndarray:
    """Helper to convert MediaPipe landmark object list to numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype="float16")


def _empty_landmarks(n_expected_landmarks: int):
    return np.full((n_expected_landmarks, 3), np.nan, dtype="float16")


def _pose_and_hands_to_arrays(
    pose_results: PoseLandmarkerResult,
    hand_results: HandLandmarkerResult,
    n_hand_landmarks=21,
    n_pose_landmarks=33,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_array = next(
        (_landmarks_to_numpy(landmarks) for landmarks in pose_results.pose_landmarks),
        None,
    )
    if pose_array is None:
        return (
            _empty_landmarks(n_pose_landmarks),
            _empty_landmarks(n_hand_landmarks),
            _empty_landmarks(n_hand_landmarks),
        )
    hand_arrays = [
        _landmarks_to_numpy(landmarks) for landmarks in hand_results.hand_landmarks
    ]
    if len(hand_arrays) < 1:
        return (
            pose_array,
            _empty_landmarks(n_hand_landmarks),
            _empty_landmarks(n_hand_landmarks),
        )
    hand_arrays = np.stack(hand_arrays, axis=0)
    wrist_dists = np.linalg.norm(
        pose_array[None, [15, 16], :2] - hand_arrays[:, [0], :2], axis=-1
    )
    hand_indices = np.argmin(wrist_dists, axis=1)
    is_left_hand, is_right_hand = (hand_indices == 0), (hand_indices == 1)
    left_hand = (
        hand_arrays[np.argmax(is_left_hand)]
        if np.any(is_left_hand)
        else _empty_landmarks(n_hand_landmarks)
    )
    right_hand = (
        hand_arrays[np.argmax(is_right_hand)]
        if np.any(is_right_hand)
        else _empty_landmarks(n_hand_landmarks)
    )
    return pose_array, left_hand, right_hand


def _face_to_array(results: FaceLandmarkerResult, n_landmarks=478) -> np.ndarray:
    landmarks = next(iter(results.face_landmarks), None)
    if landmarks is None:
        return _empty_landmarks(n_landmarks)
    return _landmarks_to_numpy(landmarks)


def extract_poses(
    video: str,
    pose_landmarker,
    hand_landmarker,
    face_landmarker=None,
    show_progress=False,
) -> dict[str, np.ndarray]:
    poses = defaultdict(lambda: [])
    for idx, (timestamp_ms, frame) in enumerate(
        iterate_video_frames_using_vidgear(video, show_progress=show_progress)
    ):
        mp_img = mp.Image(mp.ImageFormat.SRGB, frame)
        pose_results = pose_landmarker.detect_for_video(mp_img, timestamp_ms)
        hand_results = hand_landmarker.detect_for_video(mp_img, timestamp_ms)

        pose_array, left_hand_array, right_hand_array = _pose_and_hands_to_arrays(
            pose_results, hand_results
        )
        poses["pose"].append(pose_array)
        poses["left_hand"].append(left_hand_array)
        poses["right_hand"].append(right_hand_array)

        if face_landmarker:
            face_results = face_landmarker.detect_for_video(mp_img, timestamp_ms)
            face_array = _face_to_array(face_results)
            poses["face"].append(face_array)
    return {k: np.stack(v, axis=0) for k, v in poses.items()}


if __name__ == "__main__":
    landmarkers = load_landmarkers(
        {
            "hand": "C:/mediapipe/models/hand_landmarker.task",
            "pose": "C:/mediapipe/models/pose_landmarker_full.task",
            # 'face': "C:/mediapipe/models/face_landmarker.task",
        }
    )
    poses = extract_poses(
        "E:/datasets/sign-language/lsfb-cont/videos/CLSFBI0103A_S001_B.mp4",
        pose_landmarker=landmarkers["pose"],
        hand_landmarker=landmarkers["hand"],
        # face_landmarker=landmarkers['face'],
    )
    # np.save("../../notebooks/poses.npy", poses, allow_pickle=True)
