import io
import os
import tarfile
from datetime import datetime

import numpy as np


def add_file_to_tar(
        name: str,
        tar_file: tarfile.TarFile,
        data: str | bytes | np.ndarray,
):
    """ Add a file to an existing in-memory TAR archive.

    Supported data types:
    - strings
    - bytes
    - dict | list (stored as json)
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
        ...
    elif isinstance(data, bytes):
        file_data = io.BytesIO(data)
        file_size = len(data)
    elif isinstance(data, np.ndarray):
        file_data = io.BytesIO()
        np.save(file_data, data, allow_pickle=False)
        file_data.seek(0)
        file_size = file_data.getbuffer().nbytes
    else:
        raise ValueError("Data must be a file path, bytes, or numpy array.")

    file_info = tarfile.TarInfo(name=name)
    file_info.size = file_size
    file_info.mode = 0o644
    file_info.mtime = int(datetime.now().timestamp())
    tar_file.addfile(file_info, file_data)
