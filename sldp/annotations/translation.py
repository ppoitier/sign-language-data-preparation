from sldp.annotations.types import Annotations


def process_translations(all_annotations: dict[str, Annotations]) -> dict[str, Annotations]:
    for sample_id, sample_annots in all_annotations.items():
        if 'translation' in sample_annots:
            sample_annots['translation'] = sample_annots['translation'].rename(columns={'label': 'text'})
    return all_annotations