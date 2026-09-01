# Multimodal Stress & Cognitive Workload Assessment

EEG baseline pipeline for multimodal stress and cognitive workload assessment.

Current focus: Person 2 — EEG baseline pipeline (branch: `feature/eeg-baseline`).

This repository contains infrastructure for reproducible, leakage-safe EEG baseline experiments.

See [docs/eeg_baseline.md](docs/eeg_baseline.md) for the EEG data contract, validation routines,
subject-independent evaluation guidelines, and instructions to run tests and experiments.

## Project layout

- `configs/` — YAML experiment configs
- `data/` — data placeholders (no biomedical data committed)
- `src/eeg/` — EEG pipeline source code
- `notebooks/` — exploratory and example notebooks
- `tests/` — automated tests using synthetic data
- `results/` — output folders (gitignored)

## Quick commands

Create virtualenv and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run tests:

```powershell
pytest -q
```

Validation entry point example:

```powershell
python -m src.eeg.data.validation --data-dir data/processed/eeg
```

Note: No biomedical data is committed to this repository. The `data/` tree is preserved
with `.gitkeep` placeholders; real data must be provided by Person 1 according to the
data contract described in `docs/eeg_baseline.md`.
