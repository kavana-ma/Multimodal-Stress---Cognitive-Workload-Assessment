# Multimodal Methodology

## Scientific task

The target is **binary cognitive workload classification**, not stress classification. Original 0-back trials map to low workload (class 0); 2-back and 3-back map to high workload (class 1). Original class counts are reported before merging.

## Modality-specific processing

The EEG branch consumes the repository's processed epochs and reuses `src/eeg/preprocessing/feature_extraction.py`. Sampling rate, channel order, acquisition filtering, artifact handling, EOG policy, epoch definition, and units must come from verified input provenance; no second destructive preprocessing pass is made.

The fNIRS branch consumes only explicitly identified preprocessed HbO/HbR epochs. Temporal summaries are extracted per channel. Raw intensity/optical-density processing is not guessed: such data must first undergo acquisition-appropriate quality control, motion handling, modified Beer-Lambert conversion, filtering, epoching, and rejection, with the applied parameters documented.

The branches meet only after modality-specific checks. Alignment uses subject ID and trial ID, then verifies binary-label agreement. Paired subjects are discovered from data; no subject list is hard-coded.

## Models and evaluation

EEG, fNIRS, and early feature fusion compare linear SVM, RBF SVM, and Random Forest. For a fair modality comparison, paired-subject EEG, fNIRS, and fusion share identical LOSO held-out subjects. EEG using all available subjects is additionally reported as a separate baseline.

Each outer fold holds out one entire subject. SVM parameters are selected with GroupKFold inside the outer training subjects. Feature scaling is part of the model pipeline and fitted only on each outer training fold. Random Forest settings and the random seed are recorded. Feature extraction is deterministic and estimates no global dataset statistics.

Pooled out-of-fold trial metrics and subject-level LOSO metrics are both saved: accuracy, balanced accuracy, positive-class precision and recall, Macro-F1, ROC-AUC, and confusion matrices. ROC-AUC is NaN when a test fold has only one class. Subject means and standard deviations are distinct from pooled trial metrics.

## Limits

No real processed dataset is checked into this repository. Therefore this implementation has not produced empirical subject counts, workload distributions, aligned trial counts, or performance claims. The automated full-workflow test uses generated arrays only to verify data contracts, LOSO flow, and artifact creation. Deep and late fusion are not implemented in this classical baseline.