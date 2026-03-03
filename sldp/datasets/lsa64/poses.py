from sldp.poses.extract_poses import batch_extract_all_poses_from_video_dir
from sldp.poses.clean_poses import clean_all_poses_from_tars


if __name__ == '__main__':
    # batch_extract_all_poses_from_video_dir(
    #     video_dir="E:/datasets/sign-language/lsa64/videos",
    #     dest_poses_dir="E:/datasets/sign-language/lsa64/poses_raw/mediapipe",
    #     landmarker_paths={},
    #     n_workers=8,
    #     max_poses_per_tar=10_000,
    #     verbose=True,
    # )
    clean_all_poses_from_tars(
        "file:E:/datasets/sign-language/lsa64/poses_raw/mediapipe/poses_{000000..000007}.tar",
        "E:/datasets/sign-language/lsa64/poses_raw/mediapipe_cleaned.tar"
    )