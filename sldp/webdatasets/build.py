import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from sldp.entities.sign_language_sample import SignLanguageSample
from sldp.utils.tar import create_inmemory_tar, add_file_to_tar, save_inmemory_tar
from sldp.annotations.io import annotations_to_json_bytes


def add_sample_to_tar(sample: SignLanguageSample, tar):
    if sample.poses:
        for body_part, pose in sample.poses.items():
            add_file_to_tar(f"{sample.id}.pose.{body_part}.npy", tar, pose)

    if sample.annotations:
        for annot_id, annots in sample.annotations.items():
            annots: pd.DataFrame
            add_file_to_tar(
                f"{sample.id}.annotations.{annot_id}.json",
                tar,
                annotations_to_json_bytes(annots),
            )

    if sample.label:
        add_file_to_tar(
            f"{sample.id}.label.txt", tar, str(sample.label).encode("utf-8")
        )
    if sample.label_id:
        add_file_to_tar(
            f"{sample.id}.label.idx", tar, str(sample.label_id).encode("ascii")
        )

    add_file_to_tar(
        f"{sample.id}.signer.txt", tar, str(sample.signer_id).encode("utf-8")
    )
    add_file_to_tar(
        f"{sample.id}.language.txt", tar, str(sample.sign_language).encode("utf-8")
    )


def group_samples_by_signers(samples: list[SignLanguageSample], n_groups: int) -> list[list[SignLanguageSample]]:
    """Splits samples into n_groups ensuring signers do not overlap between shards."""
    missing = [sample.id for sample in samples if sample.signer_id is None]
    if missing:
        raise ValueError(
            f"Cannot group by signer: {len(missing)} sample(s) have no signer_id "
            f"(e.g. {missing[:5]}). Provide a signer_mapping when loading samples."
        )

    groups = [sample.signer_id for sample in samples]
    n_unique_signers = len(set(groups))

    if n_groups > n_unique_signers:
        raise ValueError(
            f"Cannot create {n_groups} shards with only {n_unique_signers} unique signers. "
            "Reduce n_groups."
        )

    gkf = GroupKFold(n_splits=n_groups)

    shards = []
    for _, test_idx in gkf.split(samples, groups=groups):
        shards.append([samples[idx] for idx in test_idx])

    return shards


def build_sign_language_webdataset(
    samples: list[SignLanguageSample], n_shards: int, dest_filepath: str
):
    if n_shards > 1:
        sample_groups = group_samples_by_signers(samples, n_shards)
        for i, sample_batch in tqdm(
            zip(range(n_shards), sample_groups), total=n_shards, unit="shards"
        ):
            build_sign_language_webdataset(
                sample_batch, n_shards=1, dest_filepath=dest_filepath.format(i)
            )
        return
    tar, tar_buffer = create_inmemory_tar()
    for sample in samples:
        add_sample_to_tar(sample, tar)
    save_inmemory_tar(dest_filepath, tar, tar_buffer)
