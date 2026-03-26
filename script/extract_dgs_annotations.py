from sldp.datasets.dgs.annotations import parse_all_annotations_from_elan
from sldp.datasets.dgs.glosses import categorize_dgs_gloss
from sldp.annotations.process import (
    merge_all_hand_annotations,
    remove_unannotated_samples,
    remove_empty_annotations,
    add_frame_boundaries,
)
from sldp.annotations.glosses import categorize_glosses_in_all_annotations
from sldp.annotations.translation import process_translations
from sldp.annotations.io import save_annotations_to_json, read_annotations_from_json
from sldp.annotations.vocabulary import extract_vocabulary_from_all_annotations
from sldp.utils.json import to_json

if __name__ == "__main__":
    root = "F:/datasets/sign-language/dgs"
    eaf_root = f"{root}/annotations/eaf"

    annotations_dest_filepath = f"{root}/annotations/all_annotations.json"
    issues_dest_filepath = f"{root}/annotations/all_annotation_issues.json"

    print(f"Parsing annotations from ELAN files [{eaf_root}]...")
    all_annotations, all_issues = parse_all_annotations_from_elan(eaf_root)
    print(f"Saving potential annotation issues to json [{issues_dest_filepath}]...")
    to_json(all_issues, issues_dest_filepath)
    print("Remove samples without annotations...")
    all_annotations = remove_unannotated_samples(all_annotations)
    print("Remove empty translation annotations...")
    all_annotations = remove_empty_annotations(all_annotations, ["translation"])
    print("Merging overlapping hand annotations...")
    all_annotations = merge_all_hand_annotations(all_annotations)
    print("Add frame boundaries...")
    all_annotations = add_frame_boundaries(all_annotations, framerate=50)
    print("Categorize glosses...")
    all_annotations = categorize_glosses_in_all_annotations(
        all_annotations, categorize_dgs_gloss
    )
    print("Process translations...")
    all_annotations = process_translations(all_annotations)
    print(f"Saving annotations to json [{annotations_dest_filepath}]...")
    save_annotations_to_json(annotations_dest_filepath, all_annotations)

    # all_annotations = read_annotations_from_json(annotations_dest_filepath)
    # extract_vocabulary_from_all_annotations(all_annotations, key='label')
