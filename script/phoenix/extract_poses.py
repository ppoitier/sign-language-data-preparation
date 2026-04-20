from sldp.poses.extract_poses import batch_extract_all_poses_from_video_dir

if __name__ == "__main__":
    root = "F:/datasets/sign-language/phoenix"
    batch_extract_all_poses_from_video_dir(
        video_dir=f"{root}/videos",
        dest_poses_dir=f"{root}/poses/mediapipe/raw",
        landmarker_path="C:/mediapipe/holistic_landmarker.task",
        n_workers=8,
        show_progress=False,
    )
