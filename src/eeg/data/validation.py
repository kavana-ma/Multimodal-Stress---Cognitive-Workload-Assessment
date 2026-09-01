import os
import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required EEG file: {os.path.basename(path)}")


def validate_eeg_dataset(data_dir: str) -> Dict[str, Any]:
    """Validate the processed EEG dataset in `data_dir`.

    Checks presence and basic consistency of the data contract described in docs.
    Returns a report dictionary. Raises informative errors on critical issues.
    """
    data_dir = os.path.abspath(data_dir)
    logger.info("Validating EEG dataset at %s", data_dir)

    required = ["X.npy", "y.npy", "subject_id.npy", "trial_id.npy", "metadata.csv", "channel_names.json", "label_map.json"]
    for name in required:
        _require_file(os.path.join(data_dir, name))

    X = np.load(os.path.join(data_dir, "X.npy"), allow_pickle=False)
    y = np.load(os.path.join(data_dir, "y.npy"), allow_pickle=False)
    subject_id = np.load(os.path.join(data_dir, "subject_id.npy"), allow_pickle=False)
    trial_id = np.load(os.path.join(data_dir, "trial_id.npy"), allow_pickle=False)
    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    with open(os.path.join(data_dir, "channel_names.json"), "r", encoding="utf8") as fh:
        channel_names = json.load(fh)
    with open(os.path.join(data_dir, "label_map.json"), "r", encoding="utf8") as fh:
        label_map = json.load(fh)

    if X.ndim != 3:
        raise ValueError(f"Expected EEG tensor with shape (N_epochs, N_channels, N_samples), received array with ndim={X.ndim}")

    n_epochs, n_channels, n_samples = X.shape

    if len(y) != n_epochs:
        raise ValueError("y length does not match number of EEG epochs.")
    if len(subject_id) != n_epochs:
        raise ValueError("subject_id length does not match number of EEG epochs.")
    if len(trial_id) != n_epochs:
        raise ValueError("trial_id length does not match number of EEG epochs.")

    if len(metadata) != n_epochs:
        raise ValueError("metadata.csv rows do not match number of EEG epochs.")

    if len(channel_names) != n_channels:
        raise ValueError("Number of channel names does not match channel dimension of X.")

    observed_labels = sorted(list(set(int(v) for v in np.unique(y))))
    mapped_labels = sorted([int(v) for v in label_map.keys()]) if isinstance(label_map, dict) else []
    if set(observed_labels) - set(mapped_labels):
        raise ValueError("Observed labels are inconsistent with label_map.json")

    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())

    if nan_count > 0:
        logger.warning("Found NaN values in X: %d", nan_count)
    if inf_count > 0:
        logger.warning("Found infinite values in X: %d", inf_count)

    if any([s is None or str(s) == "" for s in subject_id]):
        raise ValueError("Empty subject IDs detected.")
    if any([t is None or str(t) == "" for t in trial_id]):
        raise ValueError("Empty trial IDs detected.")

    report = {
        "n_epochs": int(n_epochs),
        "n_channels": int(n_channels),
        "n_samples": int(n_samples),
        "n_subjects": int(len(set(subject_id.tolist()))),
        "n_trials": int(len(set(trial_id.tolist()))),
        "n_classes": int(len(set(y.tolist()))),
        "class_distribution": {str(k): int((y == k).sum()) for k in sorted(set(y.tolist()))},
        "nan_count": nan_count,
        "inf_count": inf_count,
        "channel_names": channel_names,
        "label_map": label_map,
    }

    return report
