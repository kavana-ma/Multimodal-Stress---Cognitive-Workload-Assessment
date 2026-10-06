import pandas as pd
import pytest

from src.fusion.alignment import align_modalities


def test_alignment_joins_subject_and_trial_and_reports_unmatched():
    eeg = pd.DataFrame({"subject_id": ["S1", "S1", "S2"], "trial_id": ["t1", "t2", "t1"], "label": [0, 1, 0]})
    fnirs = pd.DataFrame({"subject_id": ["S1", "S2"], "trial_id": ["t1", "t2"], "label": [0, 1]})

    aligned, report = align_modalities(eeg, fnirs)

    assert aligned[["subject_id", "trial_id"]].values.tolist() == [["S1", "t1"]]
    assert aligned["eeg_label"].tolist() == [0]
    assert report["matched_trials"] == 1
    assert report["unmatched_eeg_trials"] == 2
    assert report["unmatched_fnirs_trials"] == 1


def test_alignment_rejects_duplicate_keys():
    metadata = pd.DataFrame({"subject_id": ["S1", "S1"], "trial_id": ["t1", "t1"], "label": [0, 0]})

    with pytest.raises(ValueError, match="duplicate"):
        align_modalities(metadata, metadata.iloc[:1])


def test_alignment_rejects_label_mismatch():
    eeg = pd.DataFrame({"subject_id": ["S1"], "trial_id": ["t1"], "label": [0]})
    fnirs = pd.DataFrame({"subject_id": ["S1"], "trial_id": ["t1"], "label": [1]})

    with pytest.raises(ValueError, match="labels disagree"):
        align_modalities(eeg, fnirs)