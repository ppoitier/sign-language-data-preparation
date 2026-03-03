from glob import glob
from pathlib import Path


def create_sample_index(root: str):
    for video_path in glob("**/*.mp4", root_dir=root):
        video_path = Path(video_path)
        split = str(video_path.parent.stem)
        print(video_path)

if __name__ == '__main__':
    root = "E:/datasets/sign-language/autsl"
    create_sample_index(root)
