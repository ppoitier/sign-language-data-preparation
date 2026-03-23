from typing import Callable, Optional

from sldp.entities.sign_language_sample import SignLanguageSample
from sldp.annotations.io import read_annotations_from_json
from sldp.poses.io import load_poses_from_tars


def load_continuous_samples_from_annotations(root: str, sign_language: str) -> list[SignLanguageSample]:
    # TODO: change name. We don't include 'translation' in this version.
    annotation_filepath = f"{root}/annotations/all_annotations.json"
    print("Loading annotations...")
    annotations = read_annotations_from_json(annotation_filepath)
    sample_ids = annotations.keys()
    print(f"Found {len(sample_ids)} samples.")
    print("Loading poses...")
    poses = load_poses_from_tars(f"file:{root}/poses/mediapipe/poses_linear_interpolation.tar", sample_ids)
    print("Preparing samples...")
    samples = []
    for sample_id, annots in annotations.items():
        if sample_id not in poses:
            print(f"Cannot find poses for sample {sample_id}.")
            continue
        samples.append(
            SignLanguageSample(
                id=sample_id,
                sign_language=sign_language,
                signer_id=annots['signer'],
                annotations={k: annots[k] for k in ('left_hand', 'right_hand', 'both_hands')},
                poses=poses[sample_id],
            )
        )
    print("Samples loaded.")
    return samples


def load_unannotated_samples_from_poses(
        root: str,
        sign_language: str,
        signer_mapping: Optional[Callable[[SignLanguageSample], Optional[str]]] = None,
        sample_filter: Optional[Callable[[SignLanguageSample], bool]] = None,
) -> list[SignLanguageSample]:
    print("Loading poses...")
    poses = load_poses_from_tars(f"file:{root}/poses/mediapipe/poses_linear_interpolation.tar")
    print(f"Found {len(poses)} pose files.")
    print("Preparing samples...")
    samples = []
    skipped = 0
    for sample_id, pose_data in poses.items():
        sample = SignLanguageSample(
            id=sample_id,
            sign_language=sign_language,
            signer_id=None,
            annotations=None,
            poses=pose_data,
        )
        if sample_filter and not sample_filter(sample):
            skipped += 1
            continue
        if signer_mapping:
            sample.signer_id = signer_mapping(sample)
        samples.append(sample)
    print(f"{len(samples)} samples loaded ({skipped} filtered out).")
    return samples


if __name__ == "__main__":
    root = "E:/datasets/sign-language/lsfb-cont"
    print("Annotated samples...")
    load_continuous_samples_from_annotations(root, sign_language='lsfb')
    print("Unannotated samples...")
    load_unannotated_samples_from_poses(root, sign_language='lsfb')
