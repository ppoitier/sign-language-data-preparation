from collections import defaultdict
from typing import Optional, Callable, Sequence
from pathlib import Path

import pandas as pd
from pympi import Eaf


def get_linked_videos(eaf: Eaf):
    videos = []
    for descriptor in eaf.get_linked_files():
        if not {"MEDIA_URL", "MIME_TYPE"} <= descriptor.keys():
            continue
        path_attribute = (
            "RELATIVE_MEDIA_URL" if "RELATIVE_MEDIA_URL" in descriptor else "MEDIA_URL"
        )
        video_name = Path(descriptor[path_attribute]).stem
        videos.append(video_name)
    return videos


def _annotations_to_dataframe(annotations: list[tuple[int, int, str, ...]]) -> pd.DataFrame:
    annots = pd.DataFrame([a[:3] for a in annotations], columns=["start_ms", "end_ms", "label"])
    annots = annots.sort_values("start_ms", ignore_index=True)
    annots['label'] = annots['label'].str.lower().str.strip()
    return annots


def check_signers_and_tasks_in_annotations(
    annotations_per_signer_and_task: dict,
    only_signers: set[str],
    only_tasks: set[str],
    mandatory_signers: set[str],
    mandatory_tasks: set[str],
):
    if len(annotations_per_signer_and_task) < 1:
        return ['No signer found.']
    issues = []
    missing_signers = mandatory_signers.difference(annotations_per_signer_and_task.keys())
    issues += [f"Missing signer [{signer_id}]." for signer_id in missing_signers]
    unknown_signers = set(annotations_per_signer_and_task.keys()).difference(only_signers)
    issues += [f"Unknown signer [{signer_id}]." for signer_id in unknown_signers]
    for signer_id, signer_annots in annotations_per_signer_and_task.items():
        missing_tasks = mandatory_tasks.difference(signer_annots.keys())
        issues += [f"Missing task [{task_id}] for signer [{signer_id}]." for task_id in missing_tasks]
        unknown_tasks = set(signer_annots.keys()).difference(only_tasks)
        issues += [f"Unknown task [{task_id}] for signer [{signer_id}]." for task_id in unknown_tasks]
        for task_id, task_annots in signer_annots.items():
            if task_id not in mandatory_tasks:
                continue
            if task_annots.shape[0] < 1 and task_id:
                issues += [f"No annotation found in task [{task_id}] for signer [{signer_id}]."]
            # elif task_annots.shape[0] < 5 and task_id:
            #     issues += [f"Very few annotations (less than 5) found in task [{task_id}] for signer [{signer_id}]."]
    return issues


def get_annotations_per_signer_and_task(
    eaf: Eaf,
    signer_task_from_tier: Callable,
    verbose: bool = False,
):
    annotations = defaultdict(lambda: dict())
    for tier_id, _ in eaf.tiers.items():
        tier_params = eaf.get_parameters_for_tier(tier_id)
        signer_id, task_id = signer_task_from_tier(tier_id, tier_params)
        if signer_id is None or task_id is None:
            if verbose:
                print(f"Ignore tier [{tier_id}]")
            continue
        print(f"Found tier [{tier_id}]:")
        signer_annots = _annotations_to_dataframe(
            eaf.get_annotation_data_for_tier(tier_id)
        )
        if verbose:
            print(f"---- signer [{signer_id}]")
            print(f"---- task [{task_id}]")
            print(f"---- annotations [{signer_annots.shape[0]}]")
        annotations[signer_id][task_id] = signer_annots
    return annotations
