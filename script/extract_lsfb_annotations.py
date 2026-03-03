from sldp.datasets.lsfb.annotations import parse_all_annotations_from_elan
from sldp.datasets.lsfb.glosses import categorize_gloss
from sldp.annotations.process import merge_all_hand_annotations, add_frame_boundaries
from sldp.annotations.glosses import categorize_glosses_in_all_annotations
from sldp.annotations.io import save_annotations_to_json
from sldp.utils.json import to_json


if __name__ == "__main__":
    root = "E:/datasets/sign-language/lsfb-cont"
    elan_root = f"{root}/annotations/elan"

    annotations_dest_filepath = f"{root}/annotations/all_annotations.json"
    issues_dest_filepath = f"{root}/annotations/all_annotation_issues.json"

    print(f"Parsing annotations from ELAN files [{elan_root}]...")
    all_annotations, all_issues = parse_all_annotations_from_elan(elan_root)
    print(f"Saving potential annotation issues to json [{issues_dest_filepath}]...")
    to_json(all_issues, issues_dest_filepath)
    print("Merging overlapping hand annotations...")
    all_annotations = merge_all_hand_annotations(all_annotations)
    print("Add frame boundaries...")
    all_annotations = add_frame_boundaries(all_annotations, framerate=50)
    print("Categorize glosses...")
    all_annotations = categorize_glosses_in_all_annotations(all_annotations, categorize_gloss_fn=categorize_gloss)
    print(f"Saving annotations to json [{annotations_dest_filepath}]...")
    save_annotations_to_json(annotations_dest_filepath, all_annotations)

    # TODO: talk with Adelaide about pt:lbuoy-deuxieme
    # Après discussion, lbuoy c'est un signe "complémentaire" de la main faible, donc là ce serait un signe pointé accompagné d'un lbuoy.
