# Multimodal EEG-fNIRS Cognitive Workload Assessment

This repository preserves its existing multiclass EEG baseline and adds a separate binary cognitive-workload experiment for the TU Berlin simultaneous EEG-fNIRS n-back dataset. Workload labels are not interpreted as psychological stress labels.

## Objective and task

- Class 0: 0-back, low cognitive workload.
- Class 1: 2-back or 3-back, high cognitive workload.
- Original 0-back, 2-back, and 3-back counts are reported before combining classes.
- Fusion uses only successfully aligned paired-subject trials. Alignment is by `(subject_id, trial_id)`, never row number.

## Project layout

- `src/eeg/`: existing EEG loaders, validation, features, models, splitting, and baseline.
- `src/fnirs/`: fNIRS-specific epoch feature extraction.
- `src/fusion/`: modality validation, label mapping, key alignment, LOSO training, and comparison outputs.
- `configs/eeg_baseline.yaml`: existing multiclass EEG baseline configuration.
- `configs/fnirs_baseline.yaml`: fNIRS input representation and provenance contract.
- `configs/fusion.yaml`: binary labels, EEG feature settings, models, and result paths.
- `scripts/train_eeg_baseline.py`: preserved, independently executable EEG multiclass baseline.
- `scripts/validate_modality.py`, `scripts/align_modalities.py`, `scripts/run_workload_experiments.py`: workload workflow entry points.
- `results/`: generated metrics, out-of-fold predictions, confusion matrices, plots, and reproducibility records.

## Data and preprocessing

Both processed modalities use `X.npy`, `y.npy`, `subject_id.npy`, `trial_id.npy`, `metadata.csv`, and `label_map.json`. `X` must be finite numeric `(epochs, channels, time)` data. Array rows must agree with metadata, and duplicate subject-trial keys are rejected. See [docs/data_contract.md](docs/data_contract.md).

EEG is consumed as already-processed epochs. The workload runner does not repeat filtering, artifact handling, EOG rejection, or epoching. Acquisition and preprocessing details must come from verified EEG provenance; this checkout does not contain the biomedical data to establish them. Feature extraction reuses the existing EEG implementation and configured sampling rate and frequency bands.

fNIRS is a separate branch. This runner accepts only preprocessed HbO/HbR epochs and requires `data/processed/fnirs/preprocessing_config.json` to identify `preprocessed_hbo_hbr_epochs`. It does not transform raw intensity or optical density using guessed parameters. Raw fNIRS, missing provenance, or an ambiguous representation stops validation. Features are per-channel mean, standard deviation, temporal slope, absolute peak, and area under the response.

The source label mapping is explicit in `configs/fusion.yaml`. Confirm it against the dataset codebook before use. If `label_map.json` contains semantic names, the runner also checks those names against the configured mapping. Unknown or inconsistent labels stop the run.

## Evaluation and leakage controls

The independent unit is the subject. Every model uses Leave-One-Subject-Out (LOSO), with train/test subject disjointness checked per fold. SVM parameters are selected with grouped inner folds over training subjects only. Standard scaling is fitted inside each training-fold pipeline and then applied to its held-out subject. Random Forest parameters are fixed in the config. Test-subject data are not used for model selection or scaler fitting.

Four views are evaluated: all valid EEG subjects, paired-subject EEG, paired-subject fNIRS, and early concatenation of paired EEG+fNIRS features. The latter three use identical subject folds. EEG on all subjects is a separate baseline, not a same-cohort fusion comparison.

Models are linear SVM, RBF SVM, and Random Forest. Reports include pooled out-of-fold accuracy, balanced accuracy, positive-class precision/recall, Macro-F1, ROC-AUC, and per-subject metrics. ROC-AUC is NaN when an evaluation set has one class. Deep fusion is not enabled; PyTorch currently fails to load in the available Windows environment, and the implemented classical branch is the primary experiment. See [docs/multimodal_methodology.md](docs/multimodal_methodology.md).

## Run on Windows PowerShell

From the repository root, activate the existing environment and check core dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -c "import numpy, pandas, sklearn; print('core packages OK')"
```

Validate both branches:

```powershell
python scripts/validate_modality.py --modality eeg --config configs/fusion.yaml
python scripts/validate_modality.py --modality fnirs --config configs/fusion.yaml
```

Align trials, then run EEG-only, fNIRS-only, and classical early-fusion experiments:

```powershell
python scripts/align_modalities.py --config configs/fusion.yaml
python scripts/run_workload_experiments.py --config configs/fusion.yaml
```

The pre-existing multiclass EEG baseline remains independently runnable:

```powershell
python scripts/train_eeg_baseline.py --config configs/eeg_baseline.yaml
```

Run tests:

```powershell
python -m pytest -q
```

## Results and current data availability

The experiment writes `results/eeg/`, `results/fnirs/`, `results/fusion/`, and `results/comparison/`. The comparison directory includes `modality_comparison.csv`, `per_subject_metrics.csv`, `roc_curves.png`, `summary.txt`, `config_used.yaml`, and `reproducibility.json`. Each modality directory includes metrics, out-of-fold predictions, and raw/normalized confusion matrices.

No processed EEG or fNIRS dataset is present in this checkout. Real subject/trial counts, original and binary class distributions, aligned epochs, LOSO fold count, and performance metrics have therefore not been generated. Synthetic test scores verify software behavior only and are not scientific results.

## Troubleshooting

- Missing files: populate processed modality directories according to [docs/data_contract.md](docs/data_contract.md); do not commit biomedical data.
- Label mismatch or unfamiliar codes: inspect the dataset codebook and update `labels` in `configs/fusion.yaml`; never infer labels from row order.
- fNIRS provenance error: establish whether supplied data are raw intensity, optical density, or HbO/HbR epochs and document verified processing before using this epoch-level workflow.
- PyTorch DLL error: PyTorch is optional for the classical workflow; no package reinstall is needed for these models.
- Run `python -m pytest -q` to check the current project environment.
