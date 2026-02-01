import webdataset as wds
import numpy as np

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
    tars_url: str, body_parts: tuple[str, ...] = ("pose", "left_hand", "right_hand", "face")
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


def load_poses_from_tars(tars_url: str):
    samples = list(
        wds.DataPipeline(
            wds.SimpleShardList(tars_url),
            wds.tarfile_to_samples(),
            wds.decode(),
        )
    )
    sample_to_poses = lambda s: {
        k.split(".")[1]: array for k, array in s.items() if k.startswith("poses.")
    }
    return {s["__key__"]: sample_to_poses(s) for s in samples}


def add_poses_to_tar(sample_id: str, poses: dict[str, np.ndarray], tar_file):
    for body_part, pose_seq in poses.items():
        add_file_to_tar(
            f"{sample_id}.poses.{body_part}.npy", tar_file, pose_seq.astype("float16")
        )
