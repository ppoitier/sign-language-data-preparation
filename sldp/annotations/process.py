import pandas as pd
import numpy as np

from sign_language_tools.annotations.utils.iou import (
    pairwise_temporal_intersection_over_union,
)

from sldp.annotations.types import Annotations


def merge_hand_annotations(
    left_annots: pd.DataFrame,
    right_annots: pd.DataFrame,
    iou_threshold=0.99,
) -> pd.DataFrame:
    if left_annots.empty:
        return right_annots
    if right_annots.empty:
        return left_annots

    left_times = left_annots.iloc[:, :2].values
    right_times = right_annots.iloc[:, :2].values

    left_labels = left_annots["label"].values
    right_labels = right_annots["label"].values

    iou_matrix = pairwise_temporal_intersection_over_union(left_times, right_times)
    same_label_matrix = left_labels[:, None] == right_labels[None, :]

    matches_matrix = (iou_matrix >= iou_threshold) & same_label_matrix

    left_has_match_mask = np.any(matches_matrix, axis=1)
    left_kept: pd.DataFrame = left_annots[~left_has_match_mask]

    # Keep ALL Right annotations (since they "win" the merge)
    right_kept: pd.DataFrame = right_annots
    merged_df: pd.DataFrame = pd.concat([left_kept, right_kept], ignore_index=True)
    merged_df = merged_df.sort_values(by="start_ms", ignore_index=True)
    return merged_df


def merge_all_hand_annotations(
    all_annotations: dict[str, Annotations],
) -> dict[str, Annotations]:
    for video_name, annotations in all_annotations.items():
        annotations["both_hands"] = merge_hand_annotations(
            annotations["left_hand"], annotations["right_hand"]
        )
    return all_annotations


def add_frame_boundaries(
    all_annotations: dict[str, Annotations], framerate: float
) -> dict[str, Annotations]:
    for sample_id, sample_all_annotations in all_annotations.items():
        for annot_id, sample_annotations in sample_all_annotations.items():
            if isinstance(sample_annotations, pd.DataFrame):
                factor = framerate / 1000
                sample_annotations["start_frame"] = np.floor(
                    sample_annotations.loc[:, "start_ms"].values * factor
                ).astype("int32")
                sample_annotations["end_frame"] = np.ceil(
                    sample_annotations.loc[:, "end_ms"].values * factor
                ).astype("int32")
    return all_annotations


def remove_unannotated_samples(all_annotations: dict[str, Annotations]) -> dict[str, Annotations]:
    return {
        sample_id: sample_all_annotations
        for sample_id, sample_all_annotations in all_annotations.items()
        if any(
            not df.empty
            for df in sample_all_annotations.values()
            if isinstance(df, pd.DataFrame)
        )
    }


def remove_empty_annotations(
    all_annotations: dict[str, Annotations], annotation_ids: list[str]
) -> dict[str, Annotations]:
    return {
        sample_id: {
            annot_id: annot
            for annot_id, annot in sample_all_annotations.items()
            if annot_id not in annotation_ids
            or (isinstance(annot, pd.DataFrame) and not annot.empty)
        }
        for sample_id, sample_all_annotations in all_annotations.items()
    }

if __name__ == "__main__":
    left_hand = pd.DataFrame(
        [
            [10, 20, "A"],
            [30, 40, "B"],
        ],
        columns=["start_ms", "end_ms", "label"],
    )

    right_hand = pd.DataFrame(
        [[30, 40, "B"], [30, 40, "C"], [35, 40, "D"]],
        columns=["start_ms", "end_ms", "label"],
    )

    print(merge_hand_annotations(left_hand, right_hand))
