from sldp.poses.io import load_poses_ids_from_tars
from sldp.poses.extract_poses import batch_extract_all_poses_from_video_dir


if __name__ == '__main__':
    print("Loading existing sample IDs that are already extracted...")
    tars_url = "file:E:/datasets/sign-language/lsfb-cont/poses/mediapipe/raw/poses_{000000..000402}.tar"
    pose_ids = load_poses_ids_from_tars(tars_url)
    print("Start extracting missing poses...")
    statuses = batch_extract_all_poses_from_video_dir(
        video_dir="E:/datasets/sign-language/lsfb-cont/videos",
        dest_poses_dir="E:/datasets/sign-language/lsfb-cont/poses/mediapipe_missing_videos",
        landmarker_paths={
            "hand": "/home/sign-language/weights/mediapipe/hand_landmarker.task",
            "pose": "/home/sign-language/weights/mediapipe/pose_landmarker_full.task",
            "face": "/home/sign-language/weights/mediapipe/face_landmarker.task",
        },
        max_poses_per_tar=4,
        n_workers=1,
        verbose=True,
        samples_to_skip=pose_ids,
        index_offset=403,
    )
    print("Done.")