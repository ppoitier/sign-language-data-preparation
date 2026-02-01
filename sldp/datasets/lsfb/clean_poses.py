from sldp.poses.clean_poses import clean_all_poses_from_tars


if __name__ == "__main__":
    clean_all_poses_from_tars(
        source_tar_urls="file:E:/datasets/sign-language/lsfb-cont/poses_raw/mediapipe_old/poses_000000.tar",
        dest_tar_path="E:/datasets/sign-language/lsfb-cont/poses/mediapipe/cleaned_poses.tar",
        show_progress=True,
    )
