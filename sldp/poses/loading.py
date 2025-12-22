import webdataset as wds


def load_poses_from_tars(tars_url: str):
    samples = list(wds.DataPipeline(
        wds.SimpleShardList(tars_url),
        wds.tarfile_to_samples(),
        wds.decode(),
    ))
    sample_to_poses = lambda s: {k.split('.')[1]: array for k, array in s.items() if k.startswith('poses.')}
    return {s['__key__']: sample_to_poses(s) for s in samples}
