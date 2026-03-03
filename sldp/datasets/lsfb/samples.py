from typing import Optional

from sldp.entities.sign_language_sample import SignLanguageSample
from sldp.annotations.io import read_annotations_from_json
from sldp.poses.io import load_poses_from_tars


def load_continuous_samples_from_annotations(root: str):
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
                sign_language='lsfb',
                signer_id=annots['signer'],
                annotations={k: annots[k] for k in ('left_hand', 'right_hand', 'both_hands')},
                poses=poses[sample_id],
            )
        )
    print("Samples loaded.")
    return samples


if __name__ == "__main__":
    root = "E:/datasets/sign-language/lsfb-cont"
    load_continuous_samples_from_annotations(root)
