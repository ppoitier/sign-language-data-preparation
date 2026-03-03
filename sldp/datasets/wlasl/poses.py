from sldp.poses.extract_poses import batch_extract_all_poses_from_video_dir
from sldp.poses.clean_poses import clean_all_poses_from_tars


if __name__ == '__main__':
    # batch_extract_all_poses_from_video_dir(
    #     video_dir="E:/datasets/sign-language/wlasl/videos",
    #     dest_poses_dir="E:/datasets/sign-language/wlasl/poses_raw/mediapipe",
    #     landmarker_paths={},
    #     n_workers=8,
    #     max_poses_per_tar=20_000,
    #     verbose=True,
    # )
    clean_all_poses_from_tars(
        source_tar_urls="file:E:/datasets/sign-language/wlasl/poses_raw/mediapipe/poses_{000000..000008}.tar",
        dest_tar_path="E:/datasets/sign-language/wlasl/poses/mediapipe/cleaned_poses.tar",
        show_progress=True,
    )
