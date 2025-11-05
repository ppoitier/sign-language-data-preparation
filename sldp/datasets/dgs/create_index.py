import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import pandas as pd


def create_dgs_annotated_samples_index(
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


if __name__ == "__main__":
    create_dgs_annotated_samples_index('index.csv')
