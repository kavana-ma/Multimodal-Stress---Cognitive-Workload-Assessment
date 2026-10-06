# Processed Data Contract

The workload scripts consume already-processed epochs. They do not infer labels or pair modalities by row order.

## Files per modality

Required files in `data/processed/eeg/` and `data/processed/fnirs/`:

```text
X.npy
y.npy
subject_id.npy
trial_id.npy
metadata.csv
label_map.json
```

`X.npy` is finite numeric data shaped `(N_epochs, N_channels, N_timepoints)`. The remaining arrays are one-dimensional and length `N_epochs`. `metadata.csv` must have `subject_id`, `trial_id`, and `label` columns. Its subject, trial, and label values must match each corresponding `.npy` row exactly. A `(subject_id, trial_id)` pair is unique within each modality. Optional `channel_names.json` must name every channel in order.

`label_map.json` uses original numeric source codes as keys. Source class names and the explicit source-to-binary mapping are configured in `configs/fusion.yaml`; semantic names in `label_map.json` are checked when present. Verify the mapping against the TU Berlin dataset codebook. Source codes are counted before binary conversion.

## fNIRS provenance gate

`configs/fnirs_baseline.yaml` specifies the expected representation and points to `data/processed/fnirs/preprocessing_config.json`. The provenance file must identify the actual input as `preprocessed_hbo_hbr_epochs` and document the supplied processing/acquisition details. Missing provenance or a different representation stops the run. Raw intensity/optical-density inputs are not converted with invented Beer-Lambert, motion-correction, filtering, or rejection parameters.

## Pairing and labels

EEG/fNIRS pairing is the exact intersection of `(subject_id, trial_id)`. Each aligned pair must have the same binary workload label. Duplicate keys and label conflicts stop processing. Reports preserve matched and unmatched counts and keys. Only matched trials enter paired EEG, fNIRS, or fusion evaluation.

## Sanity checks

Before model fitting, the pipeline checks dimensions and lengths, finite values, row correspondence, required columns, duplicate keys, class codes, label mapping, fNIRS provenance, matched labels, subject sets, and train/test subject separation. A critical violation raises an error and prevents training.