from sldp.datasets.lsa64.samples import load_isolated_samples_from_index
from sldp.webdatasets.build import build_sign_language_webdataset


if __name__ == "__main__":
    root = "E:/datasets/sign-language/lsa64"
    samples = load_isolated_samples_from_index(root)
    build_sign_language_webdataset(samples, n_shards=8, dest_filepath=root + "/shards/shard_{:0>6}.tar")
