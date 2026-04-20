from sldp.poses.clean_poses import clean_all_poses_from_tars

if __name__ == "__main__":
    root = "F:/datasets/sign-language/phoenix"
    clean_all_poses_from_tars(
        f"file:{root}/poses/mediapipe/raw/" + "poses_{000000..000013}.tar",
        f"{root}/poses/mediapipe/poses_linear_interpolation.tar",
    )
