from typing import Optional

import numpy as np
import pandas as pd


UPPER_BODY_IDENTIFIERS = [
    "nose",
    "neck",

    "rightShoulder",
    "rightElbow",
    "rightWrist",

    "leftShoulder",
    "leftElbow",
    "leftWrist",

    "rightEye",
    "leftEye",
    "rightEar",
    "leftEar",
]

HAND_IDENTIFIERS = [
    "wrist",

    "thumbCMC",
    "thumbMP",
    "thumbIP",
    "thumbTip",

    "indexMCP",
    "indexPIP",
    "indexDIP",
    "indexTip",

    "middleMCP",
    "middlePIP",
    "middleDIP",
    "middleTip",

    "ringMCP",
    "ringPIP",
    "ringDIP",
    "ringTip",

    "littleMCP",
    "littlePIP",
    "littleDIP",
    "littleTip",
]


def _list_str_to_array(list_str):
    return np.array(list_str[1:-1].split(","), dtype='float16')


def read_wlasl_format_csv(
        filepath: str,
        label_mapping: Optional[dict[int, str]] = None,
        left_hand_suffix = '_left',
        right_hand_suffix = '_right',
        x_coord_suffix = '_X',
        y_coord_suffix = '_Y',
) -> list[dict]:
    landmark_identifiers = {
        'upper_pose': UPPER_BODY_IDENTIFIERS,
        'left_hand': [f'{i}{left_hand_suffix}' for i in HAND_IDENTIFIERS],
        'right_hand': [f'{i}{right_hand_suffix}' for i in HAND_IDENTIFIERS],
    }
    landmark_columns = {k: {'x': [f"{i}{x_coord_suffix}" for i in identifiers], 'y': [f"{i}{y_coord_suffix}" for i in identifiers]} for k, identifiers in landmark_identifiers.items()}
    df = pd.read_csv(filepath)
    samples = []
    for idx, row in df.iterrows():
        label_id = int(row['labels'])
        sample = {
            'id': f'{idx:0>8}',
            'poses': {},
            'label_id': label_id,
        }
        if label_mapping is not None:
            sample['label'] = label_mapping[label_id]
        for region, columns in landmark_columns.items():
            x_coords = np.stack(row[columns['x']].apply(_list_str_to_array).values, axis=-1)
            y_coords = np.stack(row[columns['y']].apply(_list_str_to_array).values, axis=-1)
            coords = np.stack([x_coords, y_coords], axis=-1)  # (T, L, C)
            sample["poses"][region] = coords
        samples.append(sample)
    return samples


if __name__ == "__main__":
    read_wlasl_format_csv("E:/datasets/sign-language/wlasl/spoter/WLASL100_test_25fps.csv")
