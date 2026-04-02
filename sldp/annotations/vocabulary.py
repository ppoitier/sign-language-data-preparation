from collections import Counter

from sldp.annotations.types import Annotations


def extract_vocabulary_from_annotations(
    all_annotations: dict[str, Annotations],
    annotation_key: str = "both_hands",
    label_column: str = "lemma",
    min_occurrences: int = 1,
    max_vocabulary_size: int | None = None,
) -> dict[str, int]:
    """Build a vocabulary mapping from label strings to integer ids.

    Iterates over all annotation DataFrames, counts occurrences of each
    unique value in ``label_column``, and assigns a contiguous integer id
    to every label that meets the ``min_occurrences`` threshold.

    Args:
        all_annotations: Mapping from sample id to its annotations dict,
            where each annotations dict maps annotation keys to DataFrames.
        annotation_key: Key selecting which annotation DataFrame to use
            within each sample's annotations (e.g. ``"both_hands"``).
        label_column: Column name in the annotation DataFrame whose values
            become vocabulary entries (e.g. ``"lemma"`` or ``"gloss"``).
        min_occurrences: Minimum number of times a label must appear across
            all annotations to be included in the vocabulary. Useful for
            filtering out rare or noisy labels.
        max_vocabulary_size: Maximum vocabulary size to use. Defaults to None.

    Returns:
        A dictionary mapping each label string to a unique integer id,
        sorted by descending frequency.
    """
    counter: Counter = Counter()

    for sample_id, annotations in all_annotations.items():
        df = annotations[annotation_key]
        counter.update(df[label_column].dropna().astype(str))

    # Filter by minimum occurrences and assign contiguous ids by frequency
    vocabulary = {
        label: idx
        for idx, (label, count) in enumerate(counter.most_common(max_vocabulary_size))
        if count >= min_occurrences
    }

    return vocabulary
