from collections import Counter

from sldp.annotations.types import Annotations


def extract_vocabulary_from_all_annotations(all_annotations: dict[str, Annotations], annotation_id: str = 'both_hands', key='lemma'):
    counter = Counter()
    for annotations in all_annotations.values():
        try:
            counter += Counter(annotations[annotation_id][key])
        except KeyError:
            ... # TODO HANDLE THIS
    print(counter)
