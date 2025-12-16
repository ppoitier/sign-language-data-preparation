import os
import pandas as pd

from sldp.utils.splits import create_folds


def create_sample_index(video_dir: str, dest_index_filepath: str):
    samples = []
    for entry in os.scandir(video_dir):
        if not entry.is_file() or not entry.name.endswith(".mp4"):
            continue
        sample_id = entry.name.replace(".mp4", "")
        label_id, signer_id, _ = sample_id.split("_")
        samples.append({
            "id": sample_id,
            "class": label_id,
            "signer_id": signer_id,
        })
    df = pd.DataFrame(samples, dtype=str)
    df['class'] = df['class'].astype(int) - 1
    df.to_csv(dest_index_filepath, index=False)


def create_splits(index_path: str):
    index = pd.read_csv(index_path, dtype=str)
    sample_ids = index['id'].to_list()
    label_ids = index['class'].to_list()
    signer_ids = index['signer_id'].to_list()
    folds = create_folds(sample_ids, label_ids, signer_ids, n_folds=3)



if __name__ == '__main__':
    # create_sample_index("Z:/data/lsa64/videos", "Z:/data/lsa64/index.csv")
    create_splits("Z:/data/lsa64/index.csv")
