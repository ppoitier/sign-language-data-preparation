import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm

from sldp.samples.entity import SignLanguageSample


def isolate_signs_from_continuous_sample(
    continuous_sample: SignLanguageSample,
    annotation_key: str = "both_hands",
    label_column: str = "lemma",
    vocabulary: dict[str, int] | None = None,
) -> list[SignLanguageSample]:
    """Extract isolated sign samples from a continuous sample using its annotations.

    Each row in the annotation DataFrame identified by ``annotation_key``
    becomes one isolated :class:`SignLanguageSample`.  Boundary columns
    (``start_ms``, ``end_ms``, ``start_frame``, ``end_frame``) are copied
    directly, the ``label_column`` value becomes the sample label, and all
    remaining non-boundary columns are stored as ``linguistic_metadata``.

    If ``vocabulary`` is provided, rows whose ``label_column`` value is not
    a key in the vocabulary are skipped.  The corresponding
    ``vocabulary[label]`` value is stored as ``label_id``.

    Pose data, when present, is sliced along the temporal axis (axis 0)
    using the frame boundaries (inclusive on both ends).  A sign is
    guaranteed to span at least one frame.

    Args:
        continuous_sample: A continuous sample containing annotations and,
            optionally, pose data of shape ``(T, L, C)`` per pose key.
        annotation_key: Key into ``continuous_sample.annotations`` that
            selects which annotation DataFrame to use.
        label_column: Column name inside the annotation DataFrame that
            carries the sign label (e.g. ``"lemma"`` or ``"gloss"``).
        vocabulary: Optional mapping from label string to integer id.
            When provided, only signs whose label appears in this mapping
            are kept, and ``label_id`` is populated from it.

    Returns:
        A list of isolated :class:`SignLanguageSample` instances, one per
        valid annotation row.

    Raises:
        ValueError: If ``annotation_key`` is not found in the sample
            annotations, or if ``label_column`` is missing from the
            annotation DataFrame.
    """
    if continuous_sample.annotations is None:
        raise ValueError(
            f"Sample '{continuous_sample.id}' has no annotations."
        )

    if annotation_key not in continuous_sample.annotations:
        raise ValueError(
            f"Annotation key '{annotation_key}' not found in sample "
            f"'{continuous_sample.id}'. "
            f"Available keys: {list(continuous_sample.annotations.keys())}"
        )

    annotation_df = continuous_sample.annotations[annotation_key]

    if label_column not in annotation_df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in annotation "
            f"'{annotation_key}'. "
            f"Available columns: {list(annotation_df.columns)}"
        )

    metadata_columns = [
        col
        for col in annotation_df.columns
        if col not in {"start_ms", "end_ms", "start_frame", "end_frame"}
    ]

    isolated_samples: list[SignLanguageSample] = []

    for row_idx, row in annotation_df.iterrows():
        label = row.get(label_column)
        if pd.isna(label):
            continue
        label = str(label)
        # Filter out-of-vocabulary entries early, before pose slicing
        if vocabulary is not None and label not in vocabulary:
            continue

        start_frame, end_frame = int(row["start_frame"]), int(row["end_frame"])
        start_ms, end_ms = int(row["start_ms"]), int(row["end_ms"])

        isolated_poses: Optional[dict[str, np.ndarray]] = None
        if continuous_sample.poses is not None:
            isolated_poses = {}
            for pose_key, pose_array in continuous_sample.poses.items():
                total_frames = pose_array.shape[0]
                clamped_start = max(0, start_frame)
                # Inclusive end, i.e. end_frame is included in the slice
                clamped_end = min(end_frame + 1, total_frames)
                # A sign is always at least one frame
                clamped_end = max(clamped_end, clamped_start + 1)
                isolated_poses[pose_key] = pose_array[clamped_start:clamped_end]

        linguistic_metadata = {
            col: str(row[col])
            for col in metadata_columns
            if pd.notna(row[col])
        }
        if len(linguistic_metadata) < 1:
            linguistic_metadata = None

        sample = SignLanguageSample(
            id=f"{continuous_sample.id}_{start_ms}_{end_ms}",
            sign_language=continuous_sample.sign_language,
            signer_id=continuous_sample.signer_id,
            label=label,
            label_id=vocabulary[label] if vocabulary is not None else None,
            poses=isolated_poses,
            parent_sample_id=continuous_sample.id,
            start_ms=start_ms,
            end_ms=end_ms,
            start_frame=start_frame,
            end_frame=end_frame,
            linguistic_metadata=linguistic_metadata,
        )
        isolated_samples.append(sample)

    return isolated_samples


def isolate_signs_from_continuous_samples(
    continuous_samples: list[SignLanguageSample],
    annotation_key: str = "both_hands",
    label_column: str = "lemma",
    vocabulary: dict[str, int] | None = None,
    progress: bool = False,
) -> dict[str, list[SignLanguageSample]]:
    """Extract isolated sign samples from a list of continuous samples.

    Applies :func:`isolate_signs_from_continuous_sample` to each sample
    and flattens the results into a single list.

    Args:
        continuous_samples: A list of continuous samples, each containing
            annotations and, optionally, pose data.
        annotation_key: Key into each sample's annotations dict that
            selects which annotation DataFrame to use.
        label_column: Column name inside the annotation DataFrame that
            carries the sign label (e.g. ``"lemma"`` or ``"gloss"``).
        vocabulary: Optional mapping from label string to integer id.
            When provided, only signs whose label appears in this mapping
            are kept, and ``label_id`` is populated from it.
        progress: Whether to display a progress bar. Default to False.

    Returns:
        A flat list of isolated :class:`SignLanguageSample` instances
        gathered from all input samples.
    """
    isolated_samples: dict[str, list[SignLanguageSample]] = {}

    for continuous_sample in tqdm(continuous_samples, unit=" samples", disable=not progress):
        isolated_samples[continuous_sample.id] = isolate_signs_from_continuous_sample(
            continuous_sample=continuous_sample,
            annotation_key=annotation_key,
            label_column=label_column,
            vocabulary=vocabulary,
        )

    return isolated_samples
