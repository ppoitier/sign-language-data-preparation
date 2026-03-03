import orjson


def from_json(source_filepath: str):
    with open(source_filepath, "rb") as f:
        return orjson.loads(f.read())


def to_json(object, dest_filepath: str):
    with open(dest_filepath, "wb") as f:
        f.write(orjson.dumps(object, option=orjson.OPT_INDENT_2))


def to_json_bytes(object):
    return orjson.dumps(object, option=orjson.OPT_INDENT_2)
