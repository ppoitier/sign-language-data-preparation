from sldp.datasets.phoenix.annotations import parse_phoenix_annotations
from sldp.annotations.io import save_annotations_to_json

if __name__ == "__main__":
    root = "F:/datasets/sign-language/phoenix"
    annotations = parse_phoenix_annotations(
        alignment_path=f"{root}/annotations/automatic/train.alignment",
        classes_path=f"{root}/annotations/automatic/trainingClasses.txt",
        corpus_path=f"{root}/annotations/manual/train.corpus.csv",
    )
    save_annotations_to_json(f"{root}/annotations/all_annotations.json", annotations)
