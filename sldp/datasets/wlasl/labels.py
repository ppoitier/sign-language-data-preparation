import orjson


def create_label_mapping(raw_mapping_path: str, dest_mapping_path: str):
    with open(raw_mapping_path, "rb") as f:
        raw_mapping = orjson.loads(f.read())
    new_mapping = {sample_id: data['action'][0] for sample_id, data in raw_mapping.items()}
    with open(dest_mapping_path, "wb") as f:
        f.write(orjson.dumps(new_mapping, option=orjson.OPT_INDENT_2))


if __name__ == '__main__':
    for n_classes in [100, 300, 1000, 2000]:
        create_label_mapping(
            f"E:/datasets/sign-language/wlasl/metadata/original_mappings/nslt_{n_classes}.json",
            f"E:/datasets/sign-language/wlasl/metadata/label_mappings/asl{n_classes}.json",
        )
