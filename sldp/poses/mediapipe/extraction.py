from collections import defaultdict

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarkerOptions,
    HolisticLandmarker,
    HolisticLandmarkerResult,
    RunningMode,
)

from sldp.video.frame_iteration import iterate_video_frames_using_vidgear


def load_holistic_landmarker(model_path: str, use_gpu: bool = False):
    base_options = BaseOptions(
        model_asset_path=model_path,
        delegate=BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU,
    )
    options = HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_suppression_threshold=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.2,
        min_pose_suppression_threshold=0.5,
        min_pose_landmarks_confidence=0.2,
        min_hand_landmarks_confidence=0.2,
    )
    return HolisticLandmarker.create_from_options(options)


def _landmarks_to_array(landmarks, n_expected_landmarks: int) -> np.ndarray:
    if len(landmarks) != n_expected_landmarks:
        return np.full((n_expected_landmarks, 3), np.nan, dtype="float16")
    array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype="float16")
    return array


def extract_poses(
    video: str,
    holistic_landmarker: HolisticLandmarker,
    show_progress=False,
) -> dict[str, np.ndarray]:
    poses = defaultdict(lambda: [])
    for idx, (timestamp_ms, frame) in enumerate(
        iterate_video_frames_using_vidgear(video, show_progress=show_progress)
    ):
        mp_img = mp.Image(mp.ImageFormat.SRGB, frame)
        results: HolisticLandmarkerResult = holistic_landmarker.detect_for_video(
            mp_img, timestamp_ms
        )

        poses["pose"].append(
            _landmarks_to_array(results.pose_landmarks, n_expected_landmarks=33)
        )
        poses["left_hand"].append(
            _landmarks_to_array(results.left_hand_landmarks, n_expected_landmarks=21)
        )
        poses["right_hand"].append(
            _landmarks_to_array(results.left_hand_landmarks, n_expected_landmarks=21)
        )
        poses["face"].append(
            _landmarks_to_array(results.face_landmarks, n_expected_landmarks=478)
        )

    return {k: np.stack(v, axis=0) for k, v in poses.items()}


if __name__ == "__main__":
    # https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
    landmarker = load_holistic_landmarker("C:/mediapipe/holistic_landmarker.task")
    poses = extract_poses(
        "F:/datasets/sign-language/phoenix/example.mp4",
        holistic_landmarker=landmarker,
        show_progress=True,
    )
    print({k: v.shape for k, v in poses.items()})
