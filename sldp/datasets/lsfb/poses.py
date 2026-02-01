from sldp.poses.io import load_poses_ids_from_tars
from sldp.poses.extract_poses import batch_extract_all_poses_from_video_dir
from sldp.poses.clean_poses import clean_all_poses_from_tars


if __name__ == "__main__":
    tars_url = "file:E:/datasets/sign-language/lsfb-cont/poses_raw/mediapipe_old/poses_{000000..000291}.tar"
    pose_ids = load_poses_ids_from_tars(tars_url)
    statuses = batch_extract_all_poses_from_video_dir(
        video_dir="E:/datasets/sign-language/lsfb-cont/videos",
        dest_poses_dir="E:/datasets/sign-language/lsfb-cont/poses_raw/mediapipe_old",
        landmarker_paths={
            "hand": "/home/sign-language/weights/mediapipe/hand_landmarker.task",
            "pose": "/home/sign-language/weights/mediapipe/pose_landmarker_full.task",
            "face": "/home/sign-language/weights/mediapipe/face_landmarker.task",
        },
        max_poses_per_tar=4,
        n_workers=7,
        verbose=True,
        samples_to_skip=pose_ids,
        index_offset=292,
    )
    # statuses.to_csv(
    #     "/home/sign-language/datasets/lsfb-cont/poses_raw/statuses.csv", index=False
    # )
    # clean_all_poses_from_tars(
    #     source_tar_urls="file:E:/datasets/sign-language/lsfb-cont/poses_raw/mediapipe_old/poses_000000.tar",
    #     dest_tar_path="E:/datasets/sign-language/lsfb-cont/poses/mediapipe/cleaned_poses.tar",
    #     show_progress=True,
    # )
