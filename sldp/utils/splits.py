from sklearn.model_selection import StratifiedGroupKFold


def create_folds(
        sample_ids: list[str],
        label_ids: list[str],
        signer_ids: list[str],
        n_folds: int,
) -> list[list[str]]:
    """
    Creates k-folds with different signers for cross-validation while preserving the label distribution.

    Args:
        sample_ids: List of sample IDs.
        label_ids: List of label IDs.
        signer_ids: List of signer IDs.
        n_folds: Number of obtained folds.

    Returns:
        A list of folds, each containing sample ids.
        e.g., [[sample_1, sample_2, ...], [sample_40, sample_41, ...], ...]
    """
    # Note: shuffle=True is recommended to randomize group order
    # before splitting, which helps create more balanced folds.
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_splits = []
    for _, selected_indices in sgkf.split(X=sample_ids, y=label_ids, groups=signer_ids):
        all_splits.append([sample_ids[idx] for idx in selected_indices])
    return all_splits
