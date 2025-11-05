import numpy as np
import pandas as pd


LABELS = (
    "Opaque",
    "Red",
    "Green",
    "Yellow",
    "Bright",
    "Light-blue",
    "Colors",
    "Pink",
    "Women",
    "Enemy",
    "Son",
    "Man",
    "Away",
    "Drawer",
    "Born",
    "Learn",
    "Call",
    "Skimmer",
    "Bitter",
    "Sweet milk",
    "Milk",
    "Water",
    "Food",
    "Argentina",
    "Uruguay",
    "Country",
    "Last name",
    "Where",
    "Mock",
    "Birthday",
    "Breakfast",
    "Photo",
    "Hungry",
    "Map",
    "Coin",
    "Music",
    "Ship",
    "None",
    "Name",
    "Patience",
    "Perfume",
    "Deaf",
    "Trap",
    "Rice",
    "Barbecue",
    "Candy",
    "Chewing-gum",
    "Spaghetti",
    "Yogurt",
    "Accept",
    "Thanks",
    "Shut down",
    "Appear",
    "To land",
    "Catch",
    "Help",
    "Dance",
    "Bathe",
    "Buy",
    "Copy",
    "Run",
    "Realize",
    "Give",
    "Find",
)

POSE_BODY_IDENTIFIERS = (
    "nose",
    "neck",
    "leftShoulder",
    "leftElbow",
    "leftWrist",
    "rightShoulder",
    "rightElbow",
    "rightWrist",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
)
POSE_HAND_IDENTIFIERS = (
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
)
POSE_LEFT_HAND_IDENTIFIERS = tuple([lm_id + "_left" for lm_id in POSE_HAND_IDENTIFIERS])
POSE_RIGHT_HAND_IDENTIFIERS = tuple(
    [lm_id + "_right" for lm_id in POSE_HAND_IDENTIFIERS]
)

def _str_list_to_floats(str_list: str) -> list[float]:
    return [float(elem) for elem in str_list.strip().replace('[', '').replace(']', '').split(',')]

def _row_to_sample(row):
    body_regions = (
        ('pose', POSE_BODY_IDENTIFIERS),
        ('left_hand', POSE_LEFT_HAND_IDENTIFIERS),
        ('right_hand', POSE_RIGHT_HAND_IDENTIFIERS),
    )
    poses = dict()
    for region, identifiers in body_regions:
        data = []
        for lm_id in identifiers:
            pos_x = _str_list_to_floats(row[lm_id + "_X"])
            pos_y = _str_list_to_floats(row[lm_id + "_Y"])
            data.append([pos_x, pos_y])
        poses[region] = np.array(data).transpose([2, 0, 1])
    label_id = row['labels'] - 1
    return {
        'poses': poses,
        'label': LABELS[label_id],
        'label_id': label_id,
        'metadata': {
            'video_width': row['video_size_width'],
            'video_height': row['video_size_height'],
            'video_fps': row['video_fps'],
        }
    }

def load_data_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    print(list(df['labels']))
    # samples = df.apply(_row_to_sample, axis=1)
    # sample = samples[0]
    # print(sample)

    # lm_columns = set(sum([[lm_id + '_X', lm_id + '_Y'] for lm_id in (POSE_BODY_IDENTIFIERS + POSE_LEFT_HAND_IDENTIFIERS + POSE_RIGHT_HAND_IDENTIFIERS)], start=[]))
    # print(set(df.columns) - lm_columns)


if __name__ == "__main__":
    load_data_from_csv("E:/datasets/sign-language/lsa64/LSA64_60fps.csv")
