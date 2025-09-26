import io
import tarfile

from load_openpose import read_open_pose_tar
from sldp.utils.tar import add_file_to_tar


def convert_open_pose_tar(
        source_tar_path: str,
        dest_tar_path: str,
        show_progress=False,
        body_regions=('pose', 'left_hand', 'right_hand'),
        n_coords=3,
):
    dest_buffer = io.BytesIO()
    dest_tar = tarfile.open(fileobj=dest_buffer, mode='w')

    for sample in read_open_pose_tar(source_tar_path, show_progress=show_progress, body_regions=body_regions, n_coords=n_coords):
        for region, pose in sample.poses.items():
            add_file_to_tar(f"poses/{region}/{sample.id}.npy", dest_tar, pose)

    dest_buffer.seek(0)
    with open(dest_tar_path, 'wb') as dest_file:
        dest_file.write(dest_buffer.getvalue())
    dest_tar.close()


if __name__ == "__main__":
    convert_open_pose_tar(
        source_tar_path="E:/datasets/sign-language/how2sign/test_2D_keypoints.tar.gz",
        dest_tar_path="E:/datasets/sign-language/how2sign/test_poses_raw.tar",
        body_regions=('pose', 'left_hand', 'right_hand', 'face'),
        show_progress=True,
    )
    convert_open_pose_tar(
        source_tar_path="E:/datasets/sign-language/how2sign/val_2D_keypoints.tar.gz",
        dest_tar_path="E:/datasets/sign-language/how2sign/val_poses_raw.tar",
        body_regions=('pose', 'left_hand', 'right_hand', 'face'),
        show_progress=True,
    )
    convert_open_pose_tar(
        source_tar_path="E:/datasets/sign-language/how2sign/train_2D_keypoints.tar.gz",
        dest_tar_path="E:/datasets/sign-language/how2sign/train_poses_raw.tar",
        body_regions=('pose', 'left_hand', 'right_hand', 'face'),
        show_progress=True,
    )
