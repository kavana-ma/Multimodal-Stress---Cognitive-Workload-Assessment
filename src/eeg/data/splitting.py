from typing import List, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold


def subject_group_kfold_indices(subject_ids: np.ndarray, n_splits: int = 5, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) tuples using GroupKFold over subject_ids."""
    unique_subjects = np.unique(subject_ids)
    if n_splits > len(unique_subjects):
        raise ValueError("n_splits cannot be larger than number of unique subjects")

    # GroupKFold will split by group labels on samples
    gkf = GroupKFold(n_splits=n_splits)
    indices = []
    X_dummy = np.zeros(len(subject_ids))
    for train_idx, test_idx in gkf.split(X_dummy, groups=subject_ids):
        # verify no subject overlap
        train_subjects = set(subject_ids[train_idx].tolist())
        test_subjects = set(subject_ids[test_idx].tolist())
        assert train_subjects.isdisjoint(test_subjects)
        indices.append((train_idx, test_idx))
    return indices
