from typing import Optional, Collection

import webdataset as wds
import numpy as np
from tqdm import tqdm

from sldp.utils.tar import add_file_to_tar


def load_poses_ids_from_tars(tars_url: str):
    samples = list(
        wds.DataPipeline(
            wds.SimpleShardList(tars_url),
            wds.tarfile_to_samples(),
        )
    )
    return set([sample["__key__"] for sample in samples])


def iter_poses_from_tars(
    tars_url: str,
    body_parts: tuple[str, ...] = ("pose", "left_hand", "right_hand", "face"),
):
    iterator = wds.DataPipeline(
        wds.SimpleShardList(tars_url),
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.map(lambda s: {k: s[f"poses.{k}.npy"] for k in body_parts}),
    )
    for poses in iterator:
        sample_id = poses.pop("__key__")
        poses: dict[str, np.ndarray]
        yield sample_id, poses


def load_poses_from_tars(tars_url: str, sample_ids: Optional[Collection[str]] = None):
    pipeline_steps = [
        wds.SimpleShardList(tars_url),
        wds.tarfile_to_samples(),
    ]
    if sample_ids is not None:
        sample_ids = set(sample_ids)
        pipeline_steps.append(wds.select(lambda s: s["__key__"] in sample_ids))
    pipeline_steps.append(wds.decode())

    samples = tqdm(wds.DataPipeline(*pipeline_steps), unit='samples')
    sample_to_poses = lambda s: {
        k.split(".")[1]: array for k, array in s.items() if k.startswith("poses.")
    }
    return {s["__key__"]: sample_to_poses(s) for s in samples}


def add_poses_to_tar(sample_id: str, poses: dict[str, np.ndarray], tar_file):
    for body_part, pose_seq in poses.items():
        add_file_to_tar(
            f"{sample_id}.poses.{body_part}.npy", tar_file, pose_seq.astype("float16")
        )
