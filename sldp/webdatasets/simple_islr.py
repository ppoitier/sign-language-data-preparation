import io
import tarfile

from sldp.utils.tar import add_file_to_tar


def build_simple_islr_webdataset(samples: list[dict], dest_filepath: str):
    tar_buffer = io.BytesIO()
    tar = tarfile.open(fileobj=tar_buffer, mode="w")
    for sample in samples:
        sample_id = sample['id']
        for region, poses in sample['poses'].items():
            add_file_to_tar(f'{sample_id}.pose.{region}.npy', tar, poses)
        add_file_to_tar(f'{sample_id}.label.idx', tar, str(sample['label_id']).encode('ascii'))
    with open(dest_filepath, "wb") as file:
        file.write(tar_buffer.getvalue())
    tar_buffer.seek(0)
    tar.close()


if __name__ == "__main__":
    from sldp.csv.wlasl_format import read_wlasl_format_csv

    samples = read_wlasl_format_csv("E:/datasets/sign-language/wlasl/spoter/WLASL100_train_25fps.csv")
    build_simple_islr_webdataset(samples, "E:/datasets/sign-language/wlasl/simple_shards/asl100_train.tar")
    samples = read_wlasl_format_csv("E:/datasets/sign-language/wlasl/spoter/WLASL100_val_25fps.csv")
    build_simple_islr_webdataset(samples, "E:/datasets/sign-language/wlasl/simple_shards/asl100_val.tar")
    samples = read_wlasl_format_csv("E:/datasets/sign-language/wlasl/spoter/WLASL100_test_25fps.csv")
    build_simple_islr_webdataset(samples, "E:/datasets/sign-language/wlasl/simple_shards/asl100_test.tar")
