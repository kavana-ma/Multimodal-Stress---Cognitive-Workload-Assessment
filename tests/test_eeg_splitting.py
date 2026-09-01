import numpy as np
from src.eeg.data.splitting import subject_group_kfold_indices


def test_group_kfold_no_overlap():
    subject_ids = np.array(["S1"] * 10 + ["S2"] * 8 + ["S3"] * 6)
    folds = subject_group_kfold_indices(subject_ids, n_splits=3)
    for train_idx, test_idx in folds:
        train_subjects = set(subject_ids[train_idx].tolist())
        test_subjects = set(subject_ids[test_idx].tolist())
        assert train_subjects.isdisjoint(test_subjects)
