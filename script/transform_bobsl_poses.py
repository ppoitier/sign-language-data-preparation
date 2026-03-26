from tqdm import tqdm

from sldp.utils.tar import (
    create_inmemory_tar,
    save_inmemory_tar,
    add_file_to_tar,
    iter_tar_members,
)


def transform_chunk(source_path, dest_path):
    tar, tar_buffer = create_inmemory_tar(mode="w")
    for member, member_file in tqdm(iter_tar_members(source_path)):
        _, body_part, sample_id = member.name.replace(".npy", "").split("/")
        add_file_to_tar(f"{sample_id}.poses.{body_part}.npy", tar, member_file.read())
    save_inmemory_tar(dest_path, tar, tar_buffer)


if __name__ == "__main__":
    for i in range(57):
        transform_chunk(
            source_path=rf"F:\datasets\sign-language\bobsl\chunks\poses_{i+1}.tar",
            dest_path=rf"F:\datasets\sign-language\bobsl\poses\raw\poses_{i:0>6}.tar",
        )
