from sldp.poses.extract_poses import batch_extract_all_poses_from_video_dir
from sldp.poses.io import load_poses_ids_from_tars


if __name__ == "__main__":
    tars_url = "/home/sign-language/datasets/dgs-corpus/poses_raw/mediapipe/poses_{000000..000291}.tar"
    pose_ids = load_poses_ids_from_tars(tars_url)
    statuses = batch_extract_all_poses_from_video_dir(
        video_dir="/home/sign-language/datasets/dgs-corpus/videos",
        dest_poses_dir="/home/sign-language/datasets/dgs-corpus/poses_raw/mediapipe",
        landmarker_paths={},
        max_poses_per_tar=8,
        n_workers=23,
        verbose=True,
        samples_to_skip=pose_ids,
        index_offset=292,
    )
