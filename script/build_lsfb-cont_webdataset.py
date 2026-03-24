import re

from sldp.datasets.lsfb.samples import load_continuous_samples_from_annotations, load_unannotated_samples_from_poses
from sldp.datasets.lsfb.patterns import ID_PATTERN
from sldp.webdatasets.build import build_sign_language_webdataset


if __name__ == '__main__':
    root = "E:/datasets/sign-language/lsfb-cont"
    samples = load_continuous_samples_from_annotations(root, sign_language='lsfb')
    build_sign_language_webdataset(samples, n_shards=8, dest_filepath=root + "/shards/annotated/shard_{:0>6}.tar")

    # signer_mapping_fn = lambda sample: sample.id.split('_')[1]
    # filter_pattern = re.compile(ID_PATTERN)
    # sample_filter_fn = lambda sample: bool(filter_pattern.match(sample.id))
    # samples = load_unannotated_samples_from_poses(
    #     root,
    #     sign_language='lsfb',
    #     signer_mapping=signer_mapping_fn,
    #     sample_filter=sample_filter_fn,
    # )
    # build_sign_language_webdataset(samples, n_shards=12, dest_filepath=root + "/shards/unannotated/shard_{:0>6}.tar")
