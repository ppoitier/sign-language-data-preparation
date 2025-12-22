import orjson


def read_json_file(source_filepath: str):
    with open(source_filepath, "rb") as f:
        return orjson.loads(f.read())


def write_to_json(object, dest_filepath: str):
    with open(dest_filepath, "wb") as f:
        f.write(orjson.dumps(object, option=orjson.OPT_INDENT_2))
