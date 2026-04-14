from sldp.samples.io import load_continuous_samples_from_annotations
from sldp.webdatasets.build import build_sign_language_webdataset

from sldp.poses.io import load_poses_from_tars
from sldp.annotations.io import read_annotations_from_json

if __name__ == "__main__":
    root = "F:/datasets/sign-language/dgs"

    # poses = load_poses_from_tars(
    #     tars_url=f"file:{root}/poses/mediapipe/poses_linear_interpolation.tar"
    # )
    #
    # annotations = read_annotations_from_json(f"{root}/annotations/all_annotations.json")
    #
    # print(poses.keys())
    # print(annotations.keys())

    samples = load_continuous_samples_from_annotations(root, sign_language="dgs")
    build_sign_language_webdataset(
        samples, n_shards=8, dest_filepath=root + "/shards/shard_{:0>6}.tar"
    )
