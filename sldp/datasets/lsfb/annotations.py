import re
import pathlib

from sldp.utils.iter import iter_files_in_dir
from sldp.annotations.elan import (
    get_linked_videos,
    get_annotations_per_signer_and_task,
    check_signers_and_tasks_in_annotations,
)
from sldp.annotations.types import Annotations

from pympi.Elan import Eaf


def get_signer_id_from_video_name(video_name: str):
    signer_match = re.search(pattern=r"S(\d){2,4}", string=video_name)
    if signer_match is None:
        return None
    signer_id = int(signer_match.group().replace("S", ""))
    signer_id = f"S{signer_id:0>3}"
    return signer_id


def get_signer_task_from_tier(tier_id: str, tier_params: dict):
    tier_id = tier_id.replace(" ", "").upper()
    signer_match = re.search(pattern=r"S?(\d){1,4}", string=tier_id)
    if signer_match is None:
        return None, None
    signer_id = int(signer_match.group().replace("S", ""))
    signer_id = f"S{signer_id:0>3}"
    task = None
    if re.search(pattern="(MG)|(MAINGAUCHE)|(GAUCHE)", string=tier_id):
        task = "left_hand"
    elif re.search(pattern="(MD)|(MAINDROITE)|(DROITE)", string=tier_id):
        task = "right_hand"
    elif re.search(pattern="(TRADUCTION)|(TRADCUTION)|(TRAD)", string=tier_id):
        task = "translation"
    return signer_id, task


def parse_all_annotations_from_elan(elan_root: str) -> tuple[dict[str, Annotations], list[str]]:
    all_annotations = dict()
    all_issues = list()
    for file in iter_files_in_dir(elan_root, extensions=("eaf",)):
        eaf = Eaf(file)
        video_names, signer_ids = zip(
            *[
                (pathlib.Path(v).stem, get_signer_id_from_video_name(v))
                for v in get_linked_videos(eaf)
            ]
        )
        annots = get_annotations_per_signer_and_task(
            eaf, get_signer_task_from_tier, verbose=False
        )
        all_issues += check_signers_and_tasks_in_annotations(
            annotations_per_signer_and_task=annots,
            only_signers=set(signer_ids) - {None},
            only_tasks={"left_hand", "right_hand", "traduction"},
            mandatory_signers=set(signer_ids) - {None},
            mandatory_tasks={"left_hand", "right_hand"},
        )

        signer_id_to_video_name = {s: v for v, s in zip(video_names, signer_ids)}
        annots = {
            signer_id_to_video_name[signer_id]: {**annots, "signer": signer_id}
            for signer_id, annots in annots.items()
        }
        all_annotations.update(annots)
    return all_annotations, all_issues





# if __name__ == "__main__":
    # categorize_gloss(
    #     gloss="pt:pro1++++",
    #     variation_pattern=r"^(.*?)((\([a-z0-9]*\))?\*?(?:-(?:1|2)h)?\*?\+*)$",
    #     pt_pattern="^pt$",
    #     depictive_pattern="^ds$",
    #     buoys_pattern="^lbuoy$",
    # )

    # root = "E:/datasets/sign-language/lsfb-cont"
    #
    # annots = read_annotations_from_json(f"{root}/annotations/json/all.json")
    # annots = merge_all_hand_annotations(annots)
    #
    # categorize_glosses_in_all_annotations(
    #     annots,
    #     variation_pattern=r"^(.*?)((\([a-z0-9]*\))?\*?(?:-(?:1|2)h)?\*?\+*)$",
    #     pt_pattern="^pt$",
    #     depictive_pattern="^ds$",
    #     buoys_pattern="^lbuoy$",
    # )
    #
    # example = annots['CLSFBI0103A_S001_B']['both_hands']
    # print(example)
    # categorize_glosses(example, variation_pattern=r"(\([a-z0-9]*\))?\*?(-(1|2)h)?\*?\+*$")

    # print(list(sorted(set([a for a in example['label']]))))
