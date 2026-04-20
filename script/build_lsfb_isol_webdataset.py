from sldp.annotations.io import read_annotations_from_json
from sldp.annotations.vocabulary import extract_vocabulary_from_annotations
from sldp.samples.io import iter_samples_from_webdataset
from sldp.samples.isolation import isolate_signs_from_continuous_samples
from sldp.webdatasets.build import build_sign_language_webdataset
from sldp.video.isolation import create_clips_from_video_tar

from sldp.utils.json import from_json

from collections import Counter


if __name__ == '__main__':
    cont_root = 'F:/datasets/sign-language/lsfb-cont'
    isol_root = 'F:/datasets/sign-language/lsfb-isol'

    annotations = read_annotations_from_json(f"{cont_root}/annotations/all_annotations.json")
    # vocab = extract_vocabulary_from_annotations(annotations)

    print("Loading continuous samples...")
    samples = list(iter_samples_from_webdataset(f"file:{cont_root}/shards/annotated" + "/shard_{000000..000007}.tar"))

    for vocab_size in [500, 750, 2000, None]:
        shard_folder = 'all' if vocab_size is None else str(vocab_size)
        vocab = extract_vocabulary_from_annotations(annotations, max_vocabulary_size=vocab_size)
        print(f"Isolating samples [vocab_size={shard_folder}]...")
        samples_by_parent = isolate_signs_from_continuous_samples(samples, progress=True, vocabulary=vocab, deduplicate=True)

        print("Building isolated sign language webdataset...")
        build_sign_language_webdataset(sum(samples_by_parent.values(), start=[]), n_shards=10, dest_filepath=f"{isol_root}/shards/{shard_folder}" + "/shard_{:0>6}.tar")

    # samples_by_parent = isolate_signs_from_continuous_samples(samples, progress=True, deduplicate=True)
    # print("Extracting segment boundaries...")
    # segments = {k: [(vv.start_frame, vv.end_frame) for vv in v] for k, v in samples_by_parents.items()}

    # print("Creating clips...")
    # create_clips_from_video_tar(
    #     segments,
    #     video_dir=f"{cont_root}/videos",
    #     dest_video_tar_path=f"{isol_root}/videos.tar",
    #     fps=50,
    # )
