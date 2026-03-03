import pandas as pd

from sldp.annotations.types import Annotations


# def categorize_glosses(
#     annotations: pd.DataFrame,
#     categorize_gloss_fn,
# ) -> pd.DataFrame:
#     parsed_rows = []
#     for _, row in annotations.iterrows():
#         start_ms, end_ms, label = row["start_ms"], row["end_ms"], row['label']
#         lemma, sign_type, specifier, variation = categorize_gloss_fn(gloss=label)
#         parsed_rows.append(
#             {
#                 "start_ms": start_ms,
#                 "end_ms": end_ms,
#                 "gloss": label,
#                 "lemma": lemma,
#                 "sign_type": sign_type,
#                 "specifier": specifier,
#                 "variation": variation,
#             }
#         )
#
#     return pd.DataFrame(parsed_rows)


def categorize_glosses(
    annotations: pd.DataFrame,
    categorize_gloss_fn,
) -> pd.DataFrame:
    if annotations.shape[0] == 0:
        return (
            annotations.copy()
            .rename(columns={"label": "gloss"})
            .assign(lemma=None, sign_type=None, specifier=None, variation=None)
        )
    df = annotations.copy()
    df = df.rename(columns={"label": "gloss"})
    parsed_data = [categorize_gloss_fn(gloss=g) for g in df["gloss"]]
    df["lemma"], df["sign_type"], df["specifier"], df["variation"] = zip(*parsed_data)
    return df


def categorize_glosses_in_all_annotations(
    all_annotations: dict[str, Annotations],
    categorize_gloss_fn,
    annotation_keys=("left_hand", "right_hand", "both_hands"),
) -> dict[str, Annotations]:
    for sample_id in all_annotations:
        for key in annotation_keys:
            all_annotations[sample_id][key] = categorize_glosses(
                all_annotations[sample_id][key], categorize_gloss_fn
            )
    return all_annotations
