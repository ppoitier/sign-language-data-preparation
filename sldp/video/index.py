import orjson

from sldp.utils.tar import get_tar_index


def build_video_tar_index(
        root: str,
        video_tar_path='videos.tar',
        dest_index_path='videos.tar.index.json',
):
    tar_index = get_tar_index(f"{root}/{video_tar_path}")
    with open(f"{root}/{dest_index_path}", "wb") as f:
        f.write(orjson.dumps(tar_index, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    build_video_tar_index("E:/datasets/sign-language/lsfb-isol")
