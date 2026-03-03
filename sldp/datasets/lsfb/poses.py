import re

from sldp.poses.clean_poses import clean_all_poses_from_tars


if __name__ == "__main__":
    id_pattern = r"CLSFBI(\d){4}(A|B)_S(\d){3}_(A|B)"
    clean_all_poses_from_tars(
        source_tar_urls="file:E:/datasets/sign-language/lsfb-cont/poses/mediapipe/raw/poses_{000000..000405}.tar",
        dest_tar_path="E:/datasets/sign-language/lsfb-cont/poses/mediapipe/poses_linear_interpolation.tar",
        show_progress=True,
        filter_func=lambda sample_id, _: re.match(id_pattern, sample_id) is not None,
    )
