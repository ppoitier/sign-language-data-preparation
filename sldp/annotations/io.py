import pandas as pd

from sldp.utils.json import to_json, from_json, to_json_bytes


def annotations_to_json_bytes(annots: pd.DataFrame) -> bytes:
    return to_json_bytes(annots.replace({float("nan"): None}).to_dict(orient="records"))


def save_annotations_to_json(
    filepath: str, all_annotations: dict[str, dict[str, pd.DataFrame | str]]
):
    json_data = {}
    for video_name, video_annots in all_annotations.items():
        json_data[video_name] = {}
        for key, value in video_annots.items():
            if isinstance(value, pd.DataFrame):
                # orient="records" yields: [{'start_ms': 100, 'lemma': 'pt', ...}, {...}]
                # We replace NaNs with None so it serializes to JSON null instead of NaN
                json_data[video_name][key] = value.replace(
                    {float("nan"): None}
                ).to_dict(orient="records")
            else:
                json_data[video_name][key] = value

    to_json(json_data, filepath)


def read_annotations_from_json(
    filepath: str,
) -> dict[str, dict[str, pd.DataFrame | str]]:
    json_data = from_json(filepath)
    annots = {}
    for video_name, video_annots in json_data.items():
        annots[video_name] = {}
        for key, value in video_annots.items():
            if isinstance(value, list):
                # pandas automatically infers columns and types from the list of dicts
                annots[video_name][key] = pd.DataFrame(value)
            else:
                annots[video_name][key] = value

    return annots
