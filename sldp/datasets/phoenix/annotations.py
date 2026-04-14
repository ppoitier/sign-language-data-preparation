import pathlib

import numpy as np
import pandas as pd

# Glosses to drop entirely (silence + sentence boundaries)
DROP_GLOSSES = {"si", "__ON__", "__OFF__", "__EMOTION__"}

# Non-lexical glosses to keep with a specific sign_type
SIGN_TYPE_OVERRIDES = {
    "__PU__": "palm-up",
    # "__EMOTION__": "emotion",
}


def parse_phoenix_annotations(
    alignment_path: str | pathlib.Path,
    classes_path: str | pathlib.Path,
    corpus_path: str | pathlib.Path,
    fps: float = 25.0,
) -> dict:
    ms_per_frame = 1000.0 / fps

    # 1. classlabel -> gloss (strip trailing HMM state digits)
    classes = pd.read_csv(classes_path, sep=r"\s+")
    classes["gloss"] = classes["signstate"].str.replace(r"\d+$", "", regex=True)
    label_to_gloss = classes.set_index("classlabel")["gloss"]

    # 2. manual corpus: sample_id -> signer
    corpus = pd.read_csv(corpus_path, sep="|")

    # 3. alignment: path + label -> sample_id, frame_idx, gloss
    align = pd.read_csv(
        alignment_path,
        sep=" ",
        header=None,
        names=["path", "classlabel"],
    )
    parts = align["path"].str.split("/")
    align["sample_id"] = parts.str[-3]
    align["frame_idx"] = align["path"].str.extract(r"fn(\d+)", expand=False).astype(int)
    align["gloss"] = align["classlabel"].map(label_to_gloss)

    # 4. Drop silence and sentence boundaries
    align = align[~align["gloss"].isin(DROP_GLOSSES)]
    align = align.sort_values(["sample_id", "frame_idx"]).reset_index(drop=True)

    # 5. Vectorized segment detection. Note: we also break segments when
    # frame_idx is non-contiguous, because dropping boundary frames can
    # leave gaps within a sample that shouldn't be merged.
    sample = align["sample_id"].to_numpy()
    gloss = align["gloss"].to_numpy()
    frame = align["frame_idx"].to_numpy()
    new_segment = np.empty(len(align), dtype=bool)
    new_segment[0] = True
    new_segment[1:] = (
        (sample[1:] != sample[:-1])
        | (gloss[1:] != gloss[:-1])
        | (frame[1:] != frame[:-1] + 1)
    )
    align["segment_id"] = np.cumsum(new_segment)

    # 6. Aggregate frames per segment
    segments = align.groupby("segment_id", sort=False).agg(
        sample_id=("sample_id", "first"),
        gloss=("gloss", "first"),
        start_frame=("frame_idx", "min"),
        end_frame=("frame_idx", "max"),
    )
    segments["start_ms"] = np.round(segments["start_frame"] * ms_per_frame).astype(int)
    segments["end_ms"] = np.round((segments["end_frame"] + 1) * ms_per_frame).astype(
        int
    )
    segments["lemma"] = segments["gloss"]
    segments["sign_type"] = segments["gloss"].map(SIGN_TYPE_OVERRIDES).fillna("lexical")
    segments["specifier"] = None
    segments["variation"] = None

    segment_fields = [
        "start_ms",
        "end_ms",
        "gloss",
        "start_frame",
        "end_frame",
        "lemma",
        "sign_type",
        "specifier",
        "variation",
    ]

    # 7. Assemble result keyed by sample_id
    signer_by_sample = corpus.set_index("id")["signer"].to_dict()
    result = {}
    for sample_id, group in segments.groupby("sample_id", sort=False):
        result[sample_id] = {
            "signer": signer_by_sample.get(sample_id),
            "both_hands": group[segment_fields].reset_index(drop=True),
        }
    return result


if __name__ == "__main__":
    root = "F:/datasets/sign-language/phoenix"
    annotations = parse_phoenix_annotations(
        alignment_path=f"{root}/annotations/automatic/train.alignment",
        classes_path=f"{root}/annotations/automatic/trainingClasses.txt",
        corpus_path=f"{root}/annotations/manual/train.corpus.csv",
    )
    print(annotations["28January_2013_Monday_heute_default-17"]["signer"])
    print(annotations["28January_2013_Monday_heute_default-17"]["both_hands"])
