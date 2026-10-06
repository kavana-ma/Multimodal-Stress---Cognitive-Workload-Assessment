import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.fnirs.features import extract_fnirs_features
from src.fusion.data import apply_binary_workload_labels, load_modality_data
from src.fusion.experiment import run_experiment


def _write_modality(directory, X, y, subjects, trials):
    directory.mkdir()
    np.save(directory / "X.npy", X)
    np.save(directory / "y.npy", y)
    np.save(directory / "subject_id.npy", subjects)
    np.save(directory / "trial_id.npy", trials)
    pd.DataFrame({"subject_id": subjects, "trial_id": trials, "label": y}).to_csv(directory / "metadata.csv", index=False)
    label_map = {str(int(label)): int(label) for label in np.unique(y)}
    (directory / "label_map.json").write_text(json.dumps(label_map), encoding="utf8")


def test_modality_contract_checks_row_alignment(tmp_path):
    directory = tmp_path / "eeg"
    _write_modality(directory, np.zeros((2, 3, 8)), np.array([0, 1]), np.array(["S1", "S1"]), np.array(["t1", "t2"]))
    np.save(directory / "trial_id.npy", np.array(["t2", "t1"]))

    with pytest.raises(ValueError, match="trial_id array does not match"):
        load_modality_data(str(directory), "EEG")


def test_binary_workload_mapping_keeps_original_classes_countable():
    metadata = pd.DataFrame({"label": [0, 1, 2, 2]})
    config = {
        "source_label_names": {0: "0-back", 1: "2-back", 2: "3-back"},
        "source_to_binary": {0: 0, 1: 1, 2: 1},
    }

    result = apply_binary_workload_labels(metadata, config, "EEG")

    assert result["source_label_name"].value_counts().to_dict() == {"3-back": 2, "0-back": 1, "2-back": 1}
    assert result["binary_label"].tolist() == [0, 1, 1, 1]


def test_fnirs_features_are_finite_and_channelwise():
    X = np.arange(2 * 2 * 10, dtype=float).reshape(2, 2, 10)

    features, names = extract_fnirs_features(X)

    assert features.shape == (2, 10)
    assert len(names) == 10
    assert np.isfinite(features).all()


def test_workload_experiment_runs_paired_loso_views(tmp_path):
    subject_ids = np.repeat([f"S{idx}" for idx in range(4)], 3)
    trial_ids = np.tile(["0back", "2back", "3back"], 4)
    labels = np.tile([0, 1, 2], 4)
    rng = np.random.default_rng(17)
    eeg_dir = tmp_path / "eeg"
    fnirs_dir = tmp_path / "fnirs"
    _write_modality(eeg_dir, rng.normal(size=(12, 2, 24)), labels, subject_ids, trial_ids)
    _write_modality(fnirs_dir, rng.normal(size=(12, 2, 12)), labels, subject_ids, trial_ids)
    (fnirs_dir / "preprocessing_config.json").write_text(
        json.dumps({"representation": "preprocessed_hbo_hbr_epochs", "sampling_rate_hz": 10}), encoding="utf8"
    )
    fnirs_config_path = tmp_path / "fnirs_baseline.yaml"
    fnirs_config_path.write_text(yaml.safe_dump({
        "data_dir": str(fnirs_dir),
        "expected_representation": "preprocessed_hbo_hbr_epochs",
        "preprocessing_config_path": str(fnirs_dir / "preprocessing_config.json"),
    }), encoding="utf8")
    config = {
        "random_state": 13,
        "inner_group_folds": 2,
        "data": {"eeg_dir": str(eeg_dir), "fnirs_dir": str(fnirs_dir)},
        "fnirs_config_path": str(fnirs_config_path),
        "labels": {
            "source_label_names": {0: "0-back", 1: "2-back", 2: "3-back"},
            "source_to_binary": {0: 0, 1: 1, 2: 1},
        },
        "fnirs": {"feature_representation": "channelwise_hbo_hbr_summary"},
        "eeg": {
            "sampling_rate_hz": 100,
            "frequency_bands_hz": {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30]},
        },
        "outputs": {name: str(tmp_path / "results" / name) for name in ("eeg", "fnirs", "fusion", "comparison")},
        "models": {
            "linear_svm": {"param_grid": {"C": [1.0]}},
            "rbf_svm": {"param_grid": {"C": [1.0], "gamma": ["scale"]}},
            "random_forest": {"parameters": {"n_estimators": 5, "max_depth": 3, "min_samples_leaf": 1, "max_features": "sqrt"}},
        },
    }
    config_path = tmp_path / "fusion.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf8")

    result = run_experiment(str(config_path))

    assert result["alignment"]["matched_trials"] == 12
    assert result["alignment"]["unmatched_eeg_trials"] == 0
    assert result["folds"].groupby("view")["test_subject"].nunique().to_dict() == {
        "early_fusion": 4,
        "eeg_all": 4,
        "eeg_paired": 4,
        "fnirs_paired": 4,
    }
    assert (tmp_path / "results" / "comparison" / "modality_comparison.png").is_file()