import itertools
from typing import Optional

import pandas as pd
from pympi import Eaf


def extract_annotations_from_elan(
    elan_path: str,
    columns: Optional[tuple[str, ...]] = None,
):
    eaf = Eaf(elan_path)
    if len(eaf.tiers) < 1:
        raise ValueError(f"Empty ELAN file.")
    tier_names = {
        "a": {
            "left_hand": "Lexeme_Sign_l_A",
            "right_hand": "Lexeme_Sign_r_A",
        },
        "b": {
            "left_hand": "Lexeme_Sign_l_B",
            "right_hand": "Lexeme_Sign_r_B",
        },
    }
    annotations = {}
    for letter, hand in itertools.product(("a", "b"), ("left_hand", "right_hand")):
        tier_name = tier_names[letter][hand]
        has_tier = tier_name in eaf.tiers
        if not has_tier:
            continue
        tier = eaf.tiers[tier_name]
        if letter not in annotations:
            annotations[letter] = {"signer": tier[2]["PARTICIPANT"]}
        annotations[letter][hand] = pd.DataFrame(
            eaf.get_annotation_data_for_tier(tier_name), columns=columns
        ).to_dict("records")
    if len(annotations) == 0:
        raise ValueError("Empty ELAN file.")
    return annotations
