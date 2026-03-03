import pandas as pd
from pympi.Elan import Eaf

from sldp.utils.iter import iter_files_in_dir
from sldp.annotations.elan import (
    get_annotations_per_signer_and_task,
    check_signers_and_tasks_in_annotations,
)
from sldp.annotations.types import Annotations


def get_signer_task_from_tier(tier_id: str, tier_params: dict):
    tier_id = tier_id.strip().lower()
    signer_id = tier_params["PARTICIPANT"].strip().lower()
    task = None
    if tier_id.startswith("lexeme_sign_l"):
        task = "left_hand"
    elif tier_id.startswith("lexeme_sign_r"):
        task = "right_hand"
    elif tier_id.startswith("translation_into_english"):
        task = "translation"
    return signer_id, task


def get_signer_letter_from_tier_ids(eaf: Eaf, signer_id: str):
    for tier_id, tier in eaf.tiers.items():
        tier_params = eaf.get_parameters_for_tier(tier_id)
        if (
            "PARTICIPANT" in tier_params
            and signer_id == tier_params["PARTICIPANT"].strip().lower()
        ):
            return tier_id.strip()[-1].lower()
    raise ValueError("No signer letter found.")


def parse_all_annotations_from_elan(
    elan_root: str,
) -> tuple[dict[str, Annotations], list[str]]:
    all_annotations = dict()
    all_issues = list()
    for file in iter_files_in_dir(elan_root, extensions=("eaf",)):
        eaf = Eaf(file)
        try:
            annots = get_annotations_per_signer_and_task(
                eaf, get_signer_task_from_tier, verbose=False
            )
        except Exception as e:
            all_issues.append(f"[{file}]" + str(e))
            continue

        signer_ids = list(annots.keys())
        all_issues += check_signers_and_tasks_in_annotations(
            annotations_per_signer_and_task=annots,
            only_signers=set(signer_ids) - {None},
            only_tasks={"left_hand", "right_hand", "traduction"},
            mandatory_signers=set(signer_ids) - {None},
            mandatory_tasks={"left_hand", "right_hand"},
        )

        filename = file.stem
        signer_id_to_sample_id = {
            signer_id: f"{filename}_1{get_signer_letter_from_tier_ids(eaf, signer_id)}1"
            for signer_id in annots.keys()
        }
        annots = {
            signer_id_to_sample_id[signer_id]: {
                "left_hand": pd.DataFrame([], columns=["start_ms", "end_ms", "label"]),
                "right_hand": pd.DataFrame([], columns=["start_ms", "end_ms", "label"]),
                **annots,
                "signer": signer_id,
            }
            for signer_id, annots in annots.items()
        }
        all_annotations.update(annots)
    return all_annotations, all_issues


if __name__ == "__main__":
    elan_root = "E:/datasets/sign-language/dgs/annotations/eaf"
    parse_all_annotations_from_elan(elan_root)
