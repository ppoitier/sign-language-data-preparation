from sldp.poses.extract_poses import batch_extract_all_poses_from_video_dir


if __name__ == "__main__":
    statuses = batch_extract_all_poses_from_video_dir(
        video_dir="/home/sign-language/datasets/dgs-corpus/videos",
        dest_poses_dir="/home/sign-language/datasets/dgs-corpus/poses_raw/mediapipe",
        landmarker_paths={},
        max_poses_per_tar=8,
        n_workers=23,
        verbose=True,
    )