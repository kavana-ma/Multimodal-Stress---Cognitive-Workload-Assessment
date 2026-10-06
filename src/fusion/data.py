from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_modality_data(data_dir: str, modality: str) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load and validate one processed modality using the shared epoch contract."""
    paths = {name: os.path.join(data_dir, name) for name in ("X.npy", "y.npy", "subject_id.npy", "trial_id.npy", "metadata.csv", "label_map.json")}
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing {modality} dataset files in {data_dir}: {', '.join(missing)}")

    X = np.load(paths["X.npy"], allow_pickle=False)
    y = np.load(paths["y.npy"], allow_pickle=False)
    subjects = np.load(paths["subject_id.npy"], allow_pickle=False)
    trials = np.load(paths["trial_id.npy"], allow_pickle=False)
    metadata = pd.read_csv(paths["metadata.csv"])
    if X.ndim != 3:
        raise ValueError(f"{modality} X must have shape (epochs, channels, time), got {X.shape}.")
    n_epochs = X.shape[0]
    arrays = {"y": y, "subject_id": subjects, "trial_id": trials}
    for name, values in arrays.items():
        if values.ndim != 1 or len(values) != n_epochs:
            raise ValueError(f"{modality} {name}.npy must be one-dimensional with {n_epochs} entries.")
    if len(metadata) != n_epochs:
        raise ValueError(f"{modality} metadata rows ({len(metadata)}) do not match X epochs ({n_epochs}).")
    required = {"subject_id", "trial_id", "label"}
    if missing_columns := required - set(metadata.columns):
        raise ValueError(f"{modality} metadata is missing columns: {sorted(missing_columns)}")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError(f"{modality} X contains non-numeric, NaN, or Inf values.")

    metadata = metadata.reset_index(drop=True).copy()
    for column, values in (("subject_id", subjects), ("trial_id", trials)):
        metadata_values = metadata[column].astype(str).str.strip().to_numpy()
        array_values = pd.Series(values).astype(str).str.strip().to_numpy()
        if (metadata_values == "").any() or not np.array_equal(metadata_values, array_values):
            raise ValueError(f"{modality} {column} array does not match metadata row-by-row.")
        metadata[column] = metadata_values
    try:
        array_labels = pd.to_numeric(pd.Series(y), errors="raise").to_numpy()
        metadata_labels = pd.to_numeric(metadata["label"], errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{modality} labels must be numeric source codes.") from exc
    if not np.array_equal(array_labels, metadata_labels):
        raise ValueError(f"{modality} y.npy does not match metadata label row-by-row.")
    if not np.isfinite(array_labels).all() or not np.equal(array_labels, np.floor(array_labels)).all():
        raise ValueError(f"{modality} labels must be finite integer source codes.")
    metadata["label"] = array_labels.astype(int)
    if metadata[["subject_id", "trial_id"]].duplicated().any():
        raise ValueError(f"{modality} contains duplicate subject_id + trial_id pairs.")

    channel_names_path = os.path.join(data_dir, "channel_names.json")
    channel_names: list[str] | None = None
    if os.path.isfile(channel_names_path):
        with open(channel_names_path, "r", encoding="utf8") as handle:
            raw_names = json.load(handle)
        if not isinstance(raw_names, list) or len(raw_names) != X.shape[1]:
            raise ValueError(f"{modality} channel_names.json must contain one name per channel in X.")
        channel_names = [str(name) for name in raw_names]

    label_map_path = os.path.join(data_dir, "label_map.json")
    label_map = None
    if os.path.isfile(label_map_path):
        with open(label_map_path, "r", encoding="utf8") as handle:
            label_map = json.load(handle)
        if not isinstance(label_map, dict):
            raise ValueError(f"{modality} label_map.json must contain an object.")

    report = {
        "modality": modality,
        "n_epochs": int(n_epochs),
        "n_channels": int(X.shape[1]),
        "n_timepoints": int(X.shape[2]),
        "n_subjects": int(metadata["subject_id"].nunique()),
        "n_trials": int(metadata["trial_id"].nunique()),
        "class_distribution": {str(int(label)): int(count) for label, count in metadata["label"].value_counts().sort_index().items()},
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "channel_names": channel_names,
        "label_map": label_map,
    }
    return X, metadata, report


def apply_binary_workload_labels(
    metadata: pd.DataFrame,
    label_config: dict[str, Any],
    modality: str,
    label_map: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply an explicit, config-declared source-code mapping to binary workload."""
    source_names = {int(code): str(name) for code, name in label_config["source_label_names"].items()}
    source_to_binary = {int(code): int(label) for code, label in label_config["source_to_binary"].items()}
    observed = set(metadata["label"].astype(int).unique())
    configured = set(source_names)
    if observed != configured:
        raise ValueError(
            f"{modality} observed source labels {sorted(observed)} do not match configured labels {sorted(configured)}; "
            "inspect the supplied label metadata and update the configuration."
        )
    if label_map is not None:
        try:
            mapped_codes = {int(code) for code in label_map}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{modality} label_map.json must use original numeric source codes as keys.") from exc
        if mapped_codes != configured:
            raise ValueError(f"{modality} label_map.json codes do not match configured source labels.")
        for code, meaning in label_config["source_label_names"].items():
            mapped_meaning = label_map[str(code)] if str(code) in label_map else label_map[int(code)]
            if isinstance(mapped_meaning, str) and any(character.isalpha() for character in mapped_meaning):
                normalize = lambda value: "".join(character.lower() for character in str(value) if character.isalnum())
                if normalize(mapped_meaning) != normalize(meaning):
                    raise ValueError(
                        f"{modality} label_map assigns source code {code} to {mapped_meaning!r}, "
                        f"but configuration declares {meaning!r}."
                    )
    if set(source_to_binary) != configured or set(source_to_binary.values()) != {0, 1}:
        raise ValueError("Configuration must map every source label to binary classes 0 and 1.")
    names_by_label = metadata["label"].map(source_names)
    if names_by_label.isna().any():
        raise ValueError(f"{modality} contains source labels without a declared semantic name.")
    result = metadata.copy()
    result["source_label_name"] = names_by_label
    result["binary_label"] = result["label"].map(source_to_binary).astype(int)
    return result


def load_fnirs_provenance(fusion_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the fNIRS branch config and verify data provenance before use."""
    fnirs_config_path = fusion_config["fnirs_config_path"]
    with open(fnirs_config_path, "r", encoding="utf8") as handle:
        fnirs_config = yaml.safe_load(handle)
    configured_dir = os.path.normcase(os.path.abspath(fusion_config["data"]["fnirs_dir"]))
    branch_dir = os.path.normcase(os.path.abspath(fnirs_config["data_dir"]))
    if configured_dir != branch_dir:
        raise ValueError("fusion.yaml and fnirs_baseline.yaml point to different fNIRS data directories.")
    provenance_path = fnirs_config["preprocessing_config_path"]
    if not os.path.isfile(provenance_path):
        raise FileNotFoundError(
            f"Missing fNIRS preprocessing provenance: {provenance_path}. "
            "Record the supplied representation and processing in preprocessing_config.json before training."
        )
    with open(provenance_path, "r", encoding="utf8") as handle:
        provenance = json.load(handle)
    expected = fnirs_config["expected_representation"]
    representation = provenance.get("representation")
    if representation != expected:
        raise ValueError(
            f"Expected fNIRS representation {expected!r}, found {representation!r}; "
            "raw intensity/optical-density data require verified preprocessing before this epoch-level pipeline."
        )
    return fnirs_config, provenance