import pandas as pd

from sldp.poses.io import load_poses_from_tars
from sldp.entities.sign_language_sample import SignLanguageSample
from sldp.datasets.lsa64.labels import LABELS


def load_isolated_samples_from_index(root: str):
    sample_index = pd.read_csv(f"{root}/index.csv", dtype=str)
    sample_ids = sample_index["id"]
    print(f"Found {len(sample_ids)} samples.")
    print("Loading poses...")
    poses = load_poses_from_tars(f"file:{root}/poses/mediapipe/poses_linear_interpolation.tar", sample_ids)
    print("Preparing samples...")
    samples = []
    for _, sample_data in sample_index.iterrows():
        sample_id = sample_data['id']
        if sample_id not in poses:
            print(f"Cannot find poses for sample {sample_id}.")
            continue
        sample_label_id = int(sample_data['class'])
        samples.append(
            SignLanguageSample(
                id=sample_id,
                sign_language='lsa',
                signer_id=sample_data['signer_id'],
                label_id=sample_label_id,
                label=LABELS[sample_label_id],
                poses=poses[sample_id],
            )
        )
    print("Samples loaded.")
    return samples


if __name__ == "__main__":
    root = "E:/datasets/sign-language/lsa64"
    load_isolated_samples_from_index(root)