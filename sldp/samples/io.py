from typing import Callable, Optional
import webdataset as wds
import pandas as pd

from sldp.samples.entity import SignLanguageSample
from sldp.annotations.columns import DEFAULT_COLUMNS
from sldp.annotations.io import read_annotations_from_json
from sldp.poses.io import load_poses_from_tars


def load_continuous_samples_from_annotations(
    root: str,
    sign_language: str,
    annotation_ids=("left_hand", "right_hand", "both_hands"),
) -> list[SignLanguageSample]:
    annotation_filepath = f"{root}/annotations/all_annotations.json"
    print("Loading annotations...")
    annotations = read_annotations_from_json(annotation_filepath)
    sample_ids = annotations.keys()
    print(f"Found {len(sample_ids)} samples.")
    print("Loading poses...")
    poses = load_poses_from_tars(
        f"file:{root}/poses/mediapipe/poses_linear_interpolation.tar", sample_ids
    )
    print("Preparing samples...")
    samples = []
    for sample_id, annots in annotations.items():
        if sample_id not in poses:
            print(f"Cannot find poses for sample {sample_id}.")
            continue
        sample = SignLanguageSample(
            id=sample_id,
            sign_language=sign_language,
            signer_id=annots["signer"],
            annotations={k: annots[k] for k in annotation_ids},
            poses=poses[sample_id],
        )
        if "translation" in annots:
            sample.annotations["translation"] = annots["translation"]
        samples.append(sample)
    print("Samples loaded.")
    return samples


def load_unannotated_samples_from_poses(
    root: str,
    sign_language: str,
    signer_mapping: Optional[Callable[[SignLanguageSample], Optional[str]]] = None,
    sample_filter: Optional[Callable[[SignLanguageSample], bool]] = None,
) -> list[SignLanguageSample]:
    print("Loading poses...")
    poses = load_poses_from_tars(
        f"file:{root}/poses/mediapipe/poses_linear_interpolation.tar"
    )
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


def _get_webdataset_map_fn(body_parts: tuple[str], annotation_ids: tuple[str]):
    def _map_fn(raw_sample: dict) -> SignLanguageSample:
        sample = SignLanguageSample(
            id=raw_sample["__key__"],
            sign_language=raw_sample["language.txt"],
            signer_id=raw_sample["signer.txt"],
        )
        annotations = {
            annot_id: pd.DataFrame(
                raw_sample[f"annotations.{annot_id}.json"],
                columns=DEFAULT_COLUMNS[annot_id],
            )
            for annot_id in annotation_ids
        }
        if len(annotations) > 0:
            sample.annotations = annotations
        poses = {
            body_part: raw_sample[f"pose.{body_part}.npy"] for body_part in body_parts
        }
        if len(poses) > 0:
            sample.poses = poses
        if "label.txt" in raw_sample:
            sample.label = raw_sample["label.txt"]
        if "label.id" in raw_sample:
            sample.label_id = raw_sample["label.id"]
        return sample

    return _map_fn


def iter_samples_from_webdataset(
    tar_url: str,
    annotation_ids: tuple[str] = (
        "both_hands",
        "left_hand",
        "right_hand",
    ),
    body_parts: tuple[str] = (
        "upper_pose",
        "left_hand",
        "right_hand",
        "left_eye",
        "right_eye",
        "left_eyebrow",
        "right_eyebrow",
        "left_iris",
        "right_iris",
        "lips",
    ),
):
    iterator = wds.DataPipeline(
        wds.SimpleShardList(tar_url),
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.map(_get_webdataset_map_fn(body_parts, annotation_ids)),
    )
    for sample in iterator:
        yield sample


if __name__ == "__main__":
    for sample in iter_samples_from_webdataset(
        "file:E:/datasets/sign-language/lsfb-cont/shards/annotated/shard_000000.tar"
    ):
        print(sample.annotations["left_hand"].columns.to_list())
