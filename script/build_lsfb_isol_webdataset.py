from sldp.annotations.io import read_annotations_from_json
from sldp.annotations.vocabulary import extract_vocabulary_from_annotations
from sldp.samples.io import iter_samples_from_webdataset
from sldp.samples.isolation import isolate_signs_from_continuous_samples
from sldp.webdatasets.build import build_sign_language_webdataset
from sldp.video.isolation import create_clips_from_video_tar

from sldp.utils.json import from_json


if __name__ == '__main__':
    cont_root = 'E:/datasets/sign-language/lsfb-cont'
    isol_root = 'E:/datasets/sign-language/lsfb-isol'

    annotations = read_annotations_from_json(f"{cont_root}/annotations/all_annotations.json")
    vocab = extract_vocabulary_from_annotations(annotations)

    print("Loading continuous samples...")
    samples = list(iter_samples_from_webdataset(f"file:{cont_root}/shards/annotated" + "/shard_{000000..000007}.tar"))

    print("Isolating samples...")
    samples_by_parents = isolate_signs_from_continuous_samples(samples, progress=True)

    # print("Building isolated sign language webdataset...")
    # build_sign_language_webdataset(sum(samples_by_parents.values(), start=[]), n_shards=20, dest_filepath=f"{isol_root}/500/shards" + "/shard_{:0>6}.tar")

    print("Extracting segment boundaries...")
    segments = {k: [(vv.start_frame, vv.end_frame) for vv in v] for k, v in samples_by_parents.items()}

    print("Creating clips...")
    create_clips_from_video_tar(
        segments,
        video_dir=f"{cont_root}/videos",
        dest_video_tar_path=f"{isol_root}/videos.tar",
        fps=50,
    )
