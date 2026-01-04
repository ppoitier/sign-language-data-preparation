from pathlib import Path


def iter_files_in_dir(dir_path: str | Path, extensions: tuple[str], recursive: bool = True):
    extensions = {ext.lower() if ext.startswith('.') else f".{ext.lower()}" for ext in extensions}
    path_obj = Path(dir_path) if isinstance(dir_path, str) else dir_path
    search_pattern = "**/*" if recursive else "*"
    for file in path_obj.glob(search_pattern):
        if file.is_file() and file.suffix.lower() in extensions:
            yield file
