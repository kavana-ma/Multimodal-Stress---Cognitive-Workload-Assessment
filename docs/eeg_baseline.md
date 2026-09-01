# EEG Baseline Pipeline (Person 2)

This document describes Person 2's responsibility: establishing an engineering-grade,
reproducible EEG baseline pipeline that enforces subject-independent evaluation and
prevents data leakage.

Key points:

- Data contract: processed EEG under `data/processed/eeg/` with `X.npy`, `y.npy`,
  `subject_id.npy`, `trial_id.npy`, `metadata.csv`, `channel_names.json`, `label_map.json`.
- `X` shape: (N_epochs, N_channels, N_samples) — dimensions must be read dynamically.
- Labels: follow `label_map.json`; do not assume numeric mapping.
- Validation: use `src.eeg.data.validation.validate_eeg_dataset` to check integrity.
- Splitting: subject-grouped splitting (GroupKFold or LeaveOneGroupOut). The pipeline
  must guarantee train and test subjects are disjoint.
- Normalization: fit scalers on training subjects only; apply to test using training params.
- Models: modular API that returns `embedding, logits` from input `x`.
- Reproducibility: use `src.eeg.utils.seed.set_seed` and record environment/versions.
- Tests: automated tests use synthetic data; real EEG data must not be committed.

How to run validation (example):

```powershell
python -m src.eeg.training.train configs/eeg_baseline.yaml
```

How to run tests:

```powershell
pytest -q
```
