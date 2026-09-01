import os
import yaml
import logging
from typing import Any

import numpy as np

from ...data.validation import validate_eeg_dataset
from ...data.splitting import subject_group_kfold_indices
from ...preprocessing.normalization import LeakSafeScaler

logger = logging.getLogger(__name__)


def load_config(path: str) -> Any:
    with open(path, "r", encoding="utf8") as fh:
        return yaml.safe_load(fh)


def run_pipeline(config_path: str):
    cfg = load_config(config_path)
    data_dir = cfg["dataset"]["path"]
    report = validate_eeg_dataset(data_dir)
    logger.info("Dataset report: %s", report)

    # Example: prepare splits
    # This function is a skeleton: full training should be implemented later
    X = np.load(os.path.join(data_dir, "X.npy"))
    subject_id = np.load(os.path.join(data_dir, "subject_id.npy"))

    folds = subject_group_kfold_indices(subject_id, n_splits=cfg["evaluation"]["n_splits"], random_state=cfg["evaluation"].get("random_state", 42))
    scaler = LeakSafeScaler()
    # For the first fold, fit scaler as demonstration
    train_idx, test_idx = folds[0]
    scaler.fit(X[train_idx])
    X_train_scaled = scaler.transform(X[train_idx])
    X_test_scaled = scaler.transform(X[test_idx])
    logger.info("Prepared first fold with %d train / %d test epochs", len(train_idx), len(test_idx))
