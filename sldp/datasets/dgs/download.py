import asyncio
import requests
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd

from sldp.utils.download import download_files


def create_url_index(
        dest_filepath: str,
        url: str = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_en.html",
):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table_rows = soup.select('table.transcripts tr')
    table_rows.pop(0)
    data_index = []
    for row in table_rows:
        sample_id = row.attrs['id']
        row_data = row.find_all('td')
        eaf = (row_data[5].a or {}).get('href')
        video_a = (row_data[6].a or {}).get('href')
        video_b = (row_data[7].a or {}).get('href')
        openpose = (row_data[12].a or {}).get('href')
        data_index.append({'id': sample_id, 'eaf': eaf, 'video_a': video_a, 'video_b': video_b, 'open_pose': openpose})
    df = pd.DataFrame(data_index)
    for column in ('eaf', 'video_a', 'video_b', 'open_pose'):
        df[column] = df[column].apply(lambda path: urljoin(url, path) if path is not None else None)
    df.to_csv(dest_filepath, index=False)


def _create_download_jobs(
        url_index: pd.DataFrame,
        video_dir: str,
        eaf_dir: str,
        openpose_dir: str,
) -> list[tuple[str, str]]:
    files_to_download = []
    for _, sample_id, eaf, video_a, video_b, open_pose in url_index.itertuples():
        for video_url, video_letter in ((video_a, 'a'), (video_b, 'b')):
            if pd.isna(video_url):
                continue
            video_ext = video_url.rsplit('.', 1)[-1]
            files_to_download.append((video_url, f"{video_dir}/{sample_id}_{video_letter}.{video_ext}"))
        if not pd.isna(eaf):
            files_to_download.append((eaf, f"{eaf_dir}/{sample_id}.eaf"))
        if not pd.isna(open_pose):
            files_to_download.append((open_pose, f"{openpose_dir}/{sample_id}.json.gz"))
    return files_to_download


async def download_dgs_dataset(url_index_filepath: str, dest_dir: str):
    url_index = pd.read_csv(url_index_filepath)
    files_to_download = _create_download_jobs(
        url_index=url_index,
        video_dir=f"{dest_dir}/videos",
        eaf_dir=f"{dest_dir}/eaf",
        openpose_dir=f"{dest_dir}/openpose",
    )
    await download_files(files_to_download, verbose=True, skip_existing=True)


if __name__ == '__main__':
    asyncio.run(download_dgs_dataset(
        "index.csv",
        dest_dir="E:/datasets/sign-language/dgs-corpus",
    ))
