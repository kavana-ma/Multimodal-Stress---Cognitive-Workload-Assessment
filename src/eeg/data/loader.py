import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


def _mat_value(path: Path, variable: str) -> np.ndarray:
    values = loadmat(path, squeeze_me=True)[variable]
    if values.dtype == object:
        values = np.array([str(item.item() if hasattr(item, "item") else item) for item in values.ravel()])
    return np.asarray(values).reshape(-1)


def load_eeg_dataset(data_dir: str) -> Dict[str, Any]:
    root = Path(data_dir)
    with h5py.File(root / "X.mat", "r") as handle:
        variables = list(handle.keys())
        if variables != ["X_all"]:
            raise ValueError(f"X.mat variables are {variables}; expected one EEG variable")
        raw_shape = tuple(handle["X_all"].shape)
        y = _mat_value(root / "y.mat", "y_all").astype(int)
        if raw_shape[-1] != len(y):
            raise ValueError(f"X epoch dimension {raw_shape} does not align with y")
        X = np.asarray(handle["X_all"], dtype=np.float32).transpose(2, 1, 0)
    subjects = _mat_value(root / "subject_id.mat", "subject_id_all").astype(str)
    trials = _mat_value(root / "trial_id.mat", "trial_id_all")
    metadata = pd.read_csv(root / "metadata.csv")
    channel_names = json.loads((root / "channel_names.json").read_text(encoding="utf-8"))
    label_map = json.loads((root / "label_map.json").read_text(encoding="utf-8"))
    config = json.loads((root / "preprocessing_config.json").read_text(encoding="utf-8"))
    if not (len(y) == len(subjects) == len(trials) == len(metadata) == X.shape[0]):
        raise ValueError("X, y, subject_id, trial_id, and metadata lengths do not match")
    if metadata["subject_id"].astype(str).to_numpy().tolist() != subjects.tolist():
        raise ValueError("metadata.csv subject_id values do not align with subject_id.mat")
    if metadata["label"].to_numpy().tolist() != y.tolist():
        raise ValueError("metadata.csv labels do not align with y.mat")
    if metadata["trial_id"].to_numpy().tolist() != trials.tolist():
        raise ValueError("metadata.csv trial_id values do not align with trial_id.mat")
    if metadata.duplicated(["subject_id", "trial_id"]).any():
        raise ValueError("Duplicate (subject_id, trial_id) records detected")
    return {"X": X, "y": y, "subject_id": subjects, "trial_id": trials,
            "metadata": metadata, "channel_names": channel_names, "label_map": label_map,
            "preprocessing_config": config, "raw_X_shape": raw_shape,
            "mat_variables": {"X.mat": variables, "y.mat": ["y_all"],
            "subject_id.mat": ["subject_id_all"], "trial_id.mat": ["trial_id_all"]}}