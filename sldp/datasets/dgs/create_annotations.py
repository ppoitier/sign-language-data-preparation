import itertools
import os

import orjson

from sldp.elan.read import extract_annotations_from_elan


def create_annotations_from_eaf_files(root: str):
    all_annots = {'left_hand': dict(), 'right_hand': dict()}
    for entry in os.scandir(f"{root}/annotations/eaf"):
        sample_id, ext = entry.name.rsplit(".", 1)
        if not entry.is_file() or ext != 'eaf':
            continue
        try:
            annots = extract_annotations_from_elan(entry.path, columns=('start_ms', 'end_ms', 'gloss_en', 'gloss_de'))
        except (ValueError, KeyError) as err:
            print(f"Failed to extract annotations from {sample_id}: {err}")
            continue
        for letter, hand in itertools.product(annots, ('left_hand', 'right_hand')):
            if (letter in all_annots) and (hand in annots[letter]):
                all_annots[hand][f"{sample_id}_{letter}"] = annots[letter][hand]
    os.makedirs(f"{root}/annotations/json", exist_ok=True)
    for hand in ('left_hand', 'right_hand'):
        with open(f"{root}/annotations/json/{hand}_all_glosses.json", 'wb') as f:
            f.write(orjson.dumps(all_annots[hand]))



if __name__ == '__main__':
    create_annotations_from_eaf_files("E:/datasets/sign-language/dgs-corpus")
