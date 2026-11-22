# Next-Generation EEG-fNIRS Multimodal Model

## Executive Summary

Completed and validated a 24-fold LOSO experiment for an end-to-end dual temporal CNN with learned modality fusion. The candidate is materially different from 50:50 probability averaging because both temporal branches and the fusion head were optimized jointly on training subjects.

The candidate did not improve the primary metric: Macro-F1 0.6118 versus baseline 0.6226. Accuracy was 0.6682 versus baseline 0.6744. It was 8.18 percentage points below the 75% intermediate target and 13.18 points below 80%.

## Verified Baseline

Original saved 50:50 prediction files reproduced accuracy 0.6744, Macro-F1 0.6226, balanced accuracy 0.65625, and ROC-AUC 0.7521 across 24 folds / 648 trials. EEG-only and fNIRS-only baseline metrics are in `baseline_metrics.csv`.

## Data and Hardware Audit

Inputs are paired preprocessed temporal epochs: EEG (27, 28, 8400) and fNIRS (27, 36, 420, 2) per participant; 24 subjects and 648 trials; 9 LOW/18 HIGH trials per subject. LOW is label 0; both 2-back and 3-back are mapped to HIGH label 1. All subject/trial/label alignment checks passed.
The host exposes Intel Iris Xe graphics, but PyTorch reports CUDA unavailable; no GPU acceleration was used or claimed. PyTorch 2.6 CPU ran in a disposable environment. Peak memory was not instrumented.

## Architecture and Training

Each modality has a compact temporal convolution branch; modality embeddings concatenate into a small MLP classifier. Parameter count is 63,254. Inputs were normalized with fold-specific statistics calculated only from the 22 training subjects. Class weights were calculated from training labels. The designated validation subject selected the best epoch (maximum five epochs, early-stop patience two); test subjects were evaluated only after selection.
Measured full LOSO runtime: 770.7 seconds (12.8 minutes). Total observed training time: 538.9s over 109 epochs; mean fold wall time: 30.6s. The one-epoch pilot took 5.86s and projected 802.4s; observed whole-pipeline runtime is separately reported because the projection did not fully capture per-fold validation, data preparation, and saving overhead.

## Candidate Results

| metric | mean | std |
|---|---|---|
| accuracy | 0.66820987654321 | 0.1409182568379142 |
| macro_f1 | 0.6118227694892852 | 0.1608508770503629 |
| balanced_accuracy | 0.6516203703703703 | 0.1244901482527345 |
| low_recall | 0.6018518518518517 | 0.3019291465206654 |
| high_recall | 0.7013888888888888 | 0.2700929741912238 |
| roc_auc | 0.785493827160494 | 0.1380558905105487 |
| pr_auc | 0.8756418690674556 | 0.0818938068812157 |
| mcc | 0.323987083850487 | 0.2566283550452929 |

Paired fold differences versus baseline:

| metric | baseline_mean | candidate_mean | paired_mean_difference | improved_folds | tied_folds | worsened_folds | bootstrap_95_low | bootstrap_95_high |
|---|---|---|---|---|---|---|---|---|
| accuracy | 0.6743827160493828 | 0.66820987654321 | -0.0061728395061728 | 9 | 4 | 11 | -0.066358024691358 | 0.0524691358024691 |
| macro_f1 | 0.6226278373700233 | 0.6118227694892852 | -0.010805067880738 | 13 | 1 | 10 | -0.0945874956864565 | 0.0658735008552268 |
| balanced_accuracy | 0.6562499999999999 | 0.6516203703703703 | -0.0046296296296296 | 12 | 1 | 11 | -0.0729166666666666 | 0.0601851851851851 |
| low_recall | 0.6018518518518517 | 0.6018518518518517 | 1.8503717077085938e-17 | 11 | 2 | 11 | -0.1760416666666666 | 0.1759259259259259 |
| high_recall | 0.7106481481481483 | 0.7013888888888888 | -0.0092592592592592 | 9 | 4 | 11 | -0.1180555555555555 | 0.1041666666666666 |
| roc_auc | 0.7520576131687243 | 0.785493827160494 | 0.0334362139917695 | 16 | 1 | 7 | -0.0097736625514403 | 0.0766460905349794 |

Macro-F1 paired difference is -0.0108; improved/tied/worsened folds: 13/1/10. The 95% fold bootstrap interval is descriptive only; overlapping LOSO training sets make fold independence questionable.

## Pilot and Failures

The fold-1 pilot trained one epoch in 5.86s and achieved validation Macro-F1 0.4863; this was exploratory, not a test result. Full LOSO completed.
Final report generation initially failed because pandas' optional `tabulate` dependency was absent. This was fixed using a built-in Markdown formatter; all saved predictions and metrics were then independently recomputed and all 24 candidate checkpoints reloaded successfully. No training was repeated.

## Decision and Limitations

Keep the original 50:50 model as preferred: the new model was lower by 1.08 percentage points Macro-F1, 0.62 points accuracy, and 0.46 points balanced accuracy, although ROC-AUC increased by 0.0334. The Macro-F1 paired 95% bootstrap interval crosses zero, so the AUC increase alone does not justify replacement. Results are exploratory selection evidence on the existing 24 folds; an independent external cohort is needed for confirmation.
No architecture alternatives beyond this single candidate were tested. No raw preprocessing was rerun. CUDA/mixed precision were unavailable. No hyperparameter search used test labels.

## Saved Artifacts

- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\config.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\input_audit.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\environment.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\runtime_benchmark.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\source_hashes.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\baseline_metrics.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\candidate_fold_metrics.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\candidate_summary.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\fold_comparison.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\fold_diagnostics.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\per_subject_metrics.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\verification.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\run_log.txt`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\run_manifest.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\src\run_experiment.py`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\src\finalize_saved_run.py`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\models\compact_multimodal_cnn.py`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\tests\test_model_contract.py`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\tests\contract_test.json`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\predictions\pilot\fold_01_validation.csv`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\checkpoints`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\predictions\candidate`
- `C:\Users\Admin\Downloads\FUSION\FUSION\results\new_result\result_new_approach\training_logs`

All generated candidate artifacts reside inside this new experiment directory; original project files and checkpoints were not modified.
