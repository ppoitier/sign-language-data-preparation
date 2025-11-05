import pandas as pd
import asyncio

from sldp.utils.download import download_files


def _create_file_list(root: str, index: pd.DataFrame):
    files_to_download = []
    for _, sample_id, eaf, video_a, video_b, open_pose in index.itertuples():
        for video_url, video_letter in ((video_a, 'a'), (video_b, 'b')):
            if pd.isna(video_url):
                continue
            video_ext = video_url.rsplit('.', 1)[-1]
            files_to_download.append((video_url, f"{root}/videos/{sample_id}_{video_letter}.{video_ext}"))
        if not pd.isna(eaf):
            files_to_download.append((eaf, f"{root}/annotations/eaf/{sample_id}.eaf"))
        if not pd.isna(open_pose):
            files_to_download.append((open_pose, f"{root}/poses/openpose/{sample_id}.json.gz"))
    return files_to_download


async def download_dgs_dataset(index_filepath: str, dest_dir: str):
    index = pd.read_csv(index_filepath)
    files_to_download = _create_file_list(dest_dir, index)
    await download_files(files_to_download, verbose=True, skip_existing=True)


if __name__ == '__main__':
    asyncio.run(download_dgs_dataset(
        "index.csv",
        dest_dir="E:/datasets/sign-language/dgs-corpus",
    ))
