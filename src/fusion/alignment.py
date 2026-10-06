from __future__ import annotations

from typing import Any

import pandas as pd


ALIGNMENT_KEYS = ["subject_id", "trial_id"]


def align_modalities(eeg: pd.DataFrame, fnirs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join modality metadata by subject and trial, rejecting ambiguous matches."""
    required = set(ALIGNMENT_KEYS + ["label"])
    for name, metadata in (("EEG", eeg), ("fNIRS", fnirs)):
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"{name} metadata is missing columns: {sorted(missing)}")
        if metadata[ALIGNMENT_KEYS].isnull().any().any():
            raise ValueError(f"{name} metadata contains an empty subject_id or trial_id.")
        if metadata.duplicated(ALIGNMENT_KEYS).any():
            duplicates = metadata.loc[metadata.duplicated(ALIGNMENT_KEYS, keep=False), ALIGNMENT_KEYS]
            raise ValueError(f"{name} metadata contains duplicate subject_id + trial_id keys: {duplicates.to_dict('records')}")

    eeg_keys = pd.MultiIndex.from_frame(eeg[ALIGNMENT_KEYS])
    fnirs_keys = pd.MultiIndex.from_frame(fnirs[ALIGNMENT_KEYS])
    matched_keys = eeg_keys.intersection(fnirs_keys, sort=False)
    eeg_only = eeg_keys.difference(fnirs_keys, sort=False)
    fnirs_only = fnirs_keys.difference(eeg_keys, sort=False)

    eeg_indexed = eeg.set_index(ALIGNMENT_KEYS)
    fnirs_indexed = fnirs.set_index(ALIGNMENT_KEYS)
    matched_eeg = eeg_indexed.loc[matched_keys]
    matched_fnirs = fnirs_indexed.loc[matched_keys]
    if not matched_eeg["label"].astype(str).equals(matched_fnirs["label"].astype(str)):
        mismatches = matched_eeg["label"].astype(str) != matched_fnirs["label"].astype(str)
        examples = matched_keys[mismatches][:10].tolist()
        raise ValueError(f"EEG/fNIRS labels disagree for matched subject-trial keys: {examples}")

    aligned = pd.concat(
        [matched_eeg.add_prefix("eeg_"), matched_fnirs.drop(columns="label").add_prefix("fnirs_")],
        axis=1,
    ).reset_index()
    report = {
        "eeg_trials": int(len(eeg)),
        "fnirs_trials": int(len(fnirs)),
        "matched_trials": int(len(matched_keys)),
        "unmatched_eeg_trials": int(len(eeg_only)),
        "unmatched_fnirs_trials": int(len(fnirs_only)),
        "eeg_only_keys": [tuple(map(str, key)) for key in eeg_only.tolist()],
        "fnirs_only_keys": [tuple(map(str, key)) for key in fnirs_only.tolist()],
    }
    return aligned, report