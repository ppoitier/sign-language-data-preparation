from sldp.datasets.lsfb.samples import load_continuous_samples_from_annotations
from sldp.webdatasets.build import build_sign_language_webdataset


if __name__ == '__main__':
    root = "E:/datasets/sign-language/lsfb-cont"
    samples = load_continuous_samples_from_annotations(root)
    build_sign_language_webdataset(samples, n_shards=8, dest_filepath=root + "/shards/shard_{:0>6}.tar")


