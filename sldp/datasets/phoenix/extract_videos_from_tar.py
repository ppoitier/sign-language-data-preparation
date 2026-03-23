import re

from tqdm import tqdm

from sldp.utils.json import from_json
from sldp.video.frame_sequence import group_index_as_frame_sequences, encode_frames_to_video


FRAME_PATH_PREFIX = "phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px"
FRAME_NUM_RE = re.compile(r"fn(\d+)")


def extract_frame_number(filename: str) -> int:
    """Extract the integer frame number from a Phoenix-style PNG filename."""
    m = FRAME_NUM_RE.search(filename)
    if m is None:
        raise ValueError(f"Cannot extract frame number from: {filename}")
    return int(m.group(1))


def load_index(index_path: str):
    print("Loading index...")
    index = from_json(index_path)
    print("Filter frame features...")
    index = {k: v for k, v in index.items() if k.startswith(FRAME_PATH_PREFIX)}
    print(f"Loaded {len(index)} frames.")
    return index


def encode_all_videos(
        tar_path: str,
        all_frames: dict[str, list[tuple[str, int, int]]],
        output_directory: str,
        progress: bool = False,
):
    for group, frames in tqdm(all_frames.items(), disable=not progress, unit=" videos"):
        sample_id = group.split("/")[-2]
        encode_frames_to_video(
            tar_path=tar_path,
            frames=frames,
            output_path=f"{output_directory}/{sample_id}.mp4"
        )


if __name__ == '__main__':
    index = load_index("E:/datasets/sign-language/phoenix/phoenix-2014.v3.tar.index.json")
    groups = group_index_as_frame_sequences(index, progress=True)
    encode_all_videos(
        tar_path="E:/datasets/sign-language/phoenix/phoenix-2014.v3.tar",
        all_frames=groups,
        output_directory="E:/datasets/sign-language/phoenix/videos",
        progress=True,
    )
