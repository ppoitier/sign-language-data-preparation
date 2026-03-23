import copy
import io
import os
import tarfile
import posixpath
from datetime import datetime

from tqdm import tqdm
import numpy as np


def create_inmemory_tar(mode='w') -> tuple[tarfile.TarFile, io.BytesIO]:
    tar_buffer = io.BytesIO()
    tar = tarfile.open(fileobj=tar_buffer, mode=mode)
    return tar, tar_buffer


def save_inmemory_tar(dest_filepath: str, tar: tarfile.TarFile, tar_buffer: io.BytesIO):
    with open(dest_filepath, "wb") as file:
        file.write(tar_buffer.getvalue())
    tar_buffer.seek(0)
    tar.close()


def add_file_to_tar(
        name: str,
        tar_file: tarfile.TarFile,
        data: str | bytes | np.ndarray,
):
    """ Add a file to an existing in-memory TAR archive.

    Supported data types:
    - strings
    - bytes
    - dict | list (stored as .json)
    - numpy array (stored as .npy)

    Args:
        name (str): Name of the file in the TAR archive
        tar_file (tarfile.TarFile): Tar archive file
        data (str | bytes | np.ndarray): Data to add to the TAR archive
    """
    if isinstance(data, str):
        # If data is a file path, open and read the file
        with open(data, "rb") as f:
            file_data = io.BytesIO(f.read())
        file_size = os.path.getsize(data)
    elif isinstance(data, list) or isinstance(data, dict):
        raise NotImplementedError()  # TODO
    elif isinstance(data, bytes):
        file_data = io.BytesIO(data)
        file_size = len(data)
    elif isinstance(data, np.ndarray):
        file_data = io.BytesIO()
        np.save(file_data, data, allow_pickle=False)
        file_data.seek(0)
        file_size = file_data.getbuffer().nbytes
    else:
        raise ValueError(f"Data [{type(data)}] must be a file path, bytes, or numpy array.")

    file_info = tarfile.TarInfo(name=name)
    file_info.size = file_size
    file_info.mode = 0o644
    file_info.mtime = int(datetime.now().timestamp())
    tar_file.addfile(file_info, file_data)


def iter_tar_members(tar: tarfile.TarFile | str, recursive: bool = False):
    """
    A helper generator that yields members from a tar archive,
    handling nested tar files if specified.

    Supported tar extensions are:
      - tar
      - tar.gz
      - tgz
      - tar.bz2
    """
    if not isinstance(tar, tarfile.TarFile):
        with tarfile.open(tar, mode="r|*") as tar_obj:
            yield from iter_tar_members(tar_obj, recursive)
        return

    tar_extensions = {".tar", ".tar.gz", ".tgz", ".tar.bz2"}
    for member in tar:
        if (
                recursive and
                member.isfile() and
                any(member.name.endswith(ext) for ext in tar_extensions)
        ):
            sub_tar_stream = tar.extractfile(member)
            if not sub_tar_stream:
                continue
            with tarfile.open(fileobj=sub_tar_stream, mode="r|*") as nested_tar:
                for nested_member in iter_tar_members(nested_tar, recursive):
                    member_copy = copy.copy(nested_member)
                    member_copy.name = posixpath.join(member.name, nested_member.name)
                    yield member_copy
        else:
            yield member


def get_tar_index(tar_path: str, progress: bool = False) -> dict[str, tuple[int, int]]:
    """
        Scans a TAR file and creates a JSON index mapping filenames to
        their byte offset and size.

        Args:
            tar_path: Path to the large TAR file (e.g., "videos.tar").
            progress: If True, show the indexing progress bar.

        Returns:
            index: Dictionary mapping member names to their byte offset and size.
        """
    print(f"Creating index for {tar_path}...")
    member_index = {}
    with tarfile.open(tar_path, "r") as tar:
        it = iter(lambda: tar.next(), None)
        if progress:
            it = tqdm(it, desc="Indexing", unit=" members")
            pbar = it
            it = iter(it)
        while True:
            try:
                member = next(it)
            except StopIteration:
                break
            if member.isfile():
                member_index[member.name] = (member.offset_data, member.size)
            if progress:
                pbar.set_postfix(indexed=len(member_index))
    return member_index


def load_bytes_from_tar(tar_path: str, offset: int, size: int) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    return data


if __name__ == "__main__":
    import io

    tar_buffer = io.BytesIO()
    tar = tarfile.open(fileobj=tar_buffer, mode="w|*")
    add_file_to_tar("example.txt", tar, data="coucou".encode('ascii'))
    tar_buffer.seek(0)

    with open("example.tar", "wb") as f:
        f.write(tar_buffer.getvalue())
