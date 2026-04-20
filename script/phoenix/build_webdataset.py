from sldp.samples.io import load_continuous_samples_from_annotations
from sldp.webdatasets.build import build_sign_language_webdataset

if __name__ == "__main__":
    root = "F:/datasets/sign-language/phoenix"

    samples = load_continuous_samples_from_annotations(
        root, sign_language="dgs", annotation_ids=("both_hands",)
    )
    build_sign_language_webdataset(
        samples,
        n_shards=5,
        dest_filepath=root + "/shards/shard_{:0>6}.tar",
    )
