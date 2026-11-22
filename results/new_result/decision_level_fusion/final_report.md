# EEG-fNIRS Decision-Fusion Enhanced Experiment

**Status:** COMPLETE
**Primary protocol:** 24-fold subject-independent LOSO (22 train / 1 validation / 1 test per fold).
**Runtime:** 1000.3 s total; data loading and validation inference: 27.6 s (20-minute limit: 1200 s).
**Environment:** Python 3.12.10; Torch 2.6.0+cpu; device cpu; temporary environment only.

## Executive Summary

The saved 50:50 probability-fusion baseline was verified from all 24 held-out prediction files. Validation-only temperature scaling and a conservative threshold rule were evaluated with the same 648 test trials. Treat these candidate comparisons as exploratory: the same 24 folds informed the investigation, so no independent confirmatory test set was available.

## Results

All figures below are fold mean +/- sample standard deviation. Brier, log-loss, and ECE are lower-is-better.

| Method | Accuracy | Macro-F1 | Balanced acc. | LOW recall | HIGH recall | ROC-AUC | PR-AUC | MCC | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_50_50 | 0.6744 +/- 0.1457 | 0.6226 +/- 0.1617 | 0.6562 +/- 0.1231 | 0.6019 +/- 0.2779 | 0.7106 +/- 0.2712 | 0.7521 +/- 0.1294 | 0.8619 +/- 0.0791 | 0.3294 +/- 0.2518 | 0.2132 +/- 0.0665 | 0.6260 +/- 0.1720 | 0.2128 +/- 0.0908 |
| calibrated_equal_50_50 | 0.6682 +/- 0.1475 | 0.6136 +/- 0.1620 | 0.6516 +/- 0.1184 | 0.6019 +/- 0.2892 | 0.7014 +/- 0.2893 | 0.7495 +/- 0.1321 | 0.8597 +/- 0.0802 | 0.3273 +/- 0.2432 | 0.2117 +/- 0.0671 | 0.6186 +/- 0.1762 | 0.2121 +/- 0.0846 |
| conservative_threshold | 0.6806 +/- 0.1520 | 0.6289 +/- 0.1662 | 0.6609 +/- 0.1212 | 0.6019 +/- 0.2680 | 0.7199 +/- 0.2849 | 0.7521 +/- 0.1294 | 0.8619 +/- 0.0791 | 0.3460 +/- 0.2556 | 0.2132 +/- 0.0665 | 0.6260 +/- 0.1720 | 0.2128 +/- 0.0908 |
| calibrated_conservative_threshold | 0.6759 +/- 0.1454 | 0.6191 +/- 0.1626 | 0.6597 +/- 0.1100 | 0.6111 +/- 0.2838 | 0.7083 +/- 0.2950 | 0.7495 +/- 0.1321 | 0.8597 +/- 0.0802 | 0.3434 +/- 0.2254 | 0.2117 +/- 0.0671 | 0.6186 +/- 0.1762 | 0.2121 +/- 0.0846 |
| majority_high | 0.6667 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.6667 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.3333 +/- 0.0000 | 5.3727 +/- 0.0000 | 0.3333 +/- 0.0000 |

### Paired Fold Comparison

Differences are paired by held-out subject. The 95% percentile bootstrap interval resamples folds; LOSO folds share overlapping training cohorts, so this interval is descriptive and does not establish statistical significance.

- `calibrated_equal_50_50` vs 50:50: Macro-F1 delta -0.0090 (95% bootstrap interval -0.0290 to +0.0095); folds improved/tied/worsened 3/14/7.
- `conservative_threshold` vs 50:50: Macro-F1 delta +0.0062 (95% bootstrap interval -0.0163 to +0.0257); folds improved/tied/worsened 4/19/1.
- `calibrated_conservative_threshold` vs 50:50: Macro-F1 delta -0.0035 (95% bootstrap interval -0.0288 to +0.0197); folds improved/tied/worsened 5/12/7.

### Unimodal Controls

| Model | Accuracy | Macro-F1 | Balanced accuracy | LOW recall | HIGH recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| EEG | 0.6080 +/- 0.2052 | 0.5216 +/- 0.2236 | 0.6134 +/- 0.1662 | 0.6296 +/- 0.4153 | 0.5972 +/- 0.4064 | 0.7814 +/- 0.1733 |
| fNIRS | 0.6327 +/- 0.1030 | 0.5699 +/- 0.1145 | 0.5833 +/- 0.1023 | 0.4352 +/- 0.2096 | 0.7315 +/- 0.1668 | 0.6407 +/- 0.1454 |
| Fusion | 0.6744 +/- 0.1457 | 0.6226 +/- 0.1617 | 0.6562 +/- 0.1231 | 0.6019 +/- 0.2779 | 0.7106 +/- 0.2712 | 0.7521 +/- 0.1294 |

## Baseline and Cohort Audit

- 24 paired subjects; each has 27 aligned trials, EEG (28 x 8400), fNIRS (36 x 420 x 2). Binary labels are LOW=0 and HIGH=1, with 9/18 trials per subject.
- The original 50:50 model probabilities are the positive-class softmax probability from each modality; the saved fusion is their arithmetic average. Predictions use the original argmax rule.
- The active 50:50 summary is checked against cached predictions; the weighted-v1 directory and its proposed training script do not exist at their expected paths.
- The original 18/3/3 experiment is a separate protocol. Its saved learned representation-fusion result must not be conflated with decision fusion or LOSO.
- Source hashes and checkpoint paths/sizes are in `source_hashes.json` and `input_audit.json`; no Git commit was available.

## Separate 18/3/3 Evidence

These saved results are secondary and are not pooled with LOSO.
- `decision_equal_50_50`: accuracy 0.7160, Macro-F1 0.6976, balanced accuracy 0.7130, ROC-AUC 0.7997 (81 trials).
- `decision_equal_validation_threshold`: accuracy 0.7654, Macro-F1 0.7336, balanced accuracy 0.7315, ROC-AUC 0.7997 (81 trials).
- `decision_validation_weighted`: accuracy 0.7407, Macro-F1 0.7302, balanced accuracy 0.7593, ROC-AUC 0.8663 (81 trials).
- `existing_representation_fusion`: accuracy 0.7284, Macro-F1 0.6885, balanced accuracy 0.6852, ROC-AUC 0.7901 (81 trials).
- `existing_eeg_only`: accuracy 0.7284, Macro-F1 0.7126, balanced accuracy 0.7315, ROC-AUC 0.8786 (81 trials).
- `existing_fnirs_only`: accuracy 0.5556, Macro-F1 0.5297, balanced accuracy 0.5370, ROC-AUC 0.5775 (81 trials).

## Diagnostic Findings

- **High / Confirmed: Class imbalance** Each subject has 9 LOW and 18 HIGH trials; majority-HIGH reference is 66.67% accuracy. Fix: Added class-wise metrics and majority baseline; no training change made.
- **Moderate / Confirmed: Validation threshold instability** Prior broad threshold tuning lowered LOSO Macro-F1; this run used a three-value grid plus +0.05 F1 and balanced-accuracy guards. Fix: Conservative validation-only threshold rule; defaults retained unless guard passes.
- **Moderate / Limitation: Calibration sample size** One validation subject contributes 27 trials per fold; temperature is selected from five fixed values. Fix: No isotonic calibration; calibrated results remain exploratory.
- **Low / Confirmed missing: Weighted v1 provenance** Expected directory exists=False; expected training script exists=False. Fix: Marked unavailable; prior v2 weighting artifacts separately recorded.
- **High / Verified: Checkpoint normalization and fold map** Each fold-specific checkpoint is loaded by fold ID and its stored training mean/std are applied to validation data. Fix: No source checkpoint or model changes.
- **High / Verified: Trial/label alignment** All baseline prediction rows were checked against MAT trial IDs and labels after each subject was loaded as a validation subject. Fix: Mismatches would stop the run; all 648 rows aligned.
- **Diagnostic / Measured: Cross-modal error correlation** Per-fold correlation and joint error counts are saved in prediction_diagnostics.csv. Fix: No change applied.

## Decision and Limitations

The 50:50 fusion remains the preferred control unless a candidate's mean LOSO Macro-F1 improves under independent confirmation. Validation calibration uses one 27-trial held-out validation subject per fold, so temperature estimates are noisy and candidate results are exploratory. Conservative threshold movement requires at least +0.05 validation Macro-F1 with no balanced-accuracy decline; otherwise the default 0.50 threshold is kept.

Regularized logistic meta-fusion was skipped because no independent training-subject out-of-fold modality probabilities were available. No optional architecture was trained: the goal was the no-retraining primary decision-fusion experiment, and training a new subject-independent model was not justified by the available saved features or runtime evidence. Raw preprocessing and original encoders were not rerun.

The Python environment lacked NumPy, h5py, pandas, scikit-learn, and Torch. A disposable Python 3.12/Torch 2.6 CPU environment was used; the global environment and requirements file were not changed. Total runtime was 1000.3 seconds, and validation data loading/inference took 27.6 seconds.

See `config.json`, `environment.json`, `input_audit.json`, `source_hashes.json`, `selected_parameters.csv`, `candidate_fold_metrics.csv`, `candidate_summary.csv`, `fold_comparison.csv`, `prediction_diagnostics.csv`, and `diagnostic_findings.csv` for the reproducibility record.


## Previous Validation-Only Run

The earlier run at `C:\Users\Admin\Downloads\FUSION\FUSION\results\decision_level_fusion_v2_20261013_114310` reported threshold-only Macro-F1 0.5903 and weighted-fusion Macro-F1 0.5852 versus 0.6226 for 50:50. This run tests calibration and a narrower, guarded threshold decision rather than repeating the earlier broad threshold search.

## Exact Output Paths

- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\config.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\input_audit.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\environment.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\source_hashes.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\baseline_metrics.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\candidate_fold_metrics.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\candidate_summary.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\fold_comparison.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\selected_parameters.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\prediction_diagnostics.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\diagnostic_findings.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\run_log.txt`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\run_manifest.json`
