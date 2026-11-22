# FUSION Project Operating Manual

## Scope and important verification result

This repository does **not** contain one single shell script or Python entry point that automatically runs every stage. The project is a collection of independent pipelines. The primary manual pipeline is the **24-fold LOSO representation-level pipeline** in `scripts/01_verify_data.py` through `scripts/10_generate_final_report.py`.

This README distinguishes:

1. **Primary LOSO baseline pipeline:** EEG, fNIRS, and representation-level fusion.
2. **Final LOSO pipeline:** tuned representation-level final model.
3. **Decision-level/late-fusion pipeline:** combines saved EEG and fNIRS probabilities.
4. **Enhanced decision-fusion experiment:** explores calibration and guarded thresholds without retraining encoders.
5. **Alternative 18/3/3 split pipeline:** independent fixed split experiment.
6. **Additional architectures:** attention, dual-chromophore, cross-modal attention, and gated-fusion experiments.

This README does **not** claim that a result exists if the corresponding script was skipped. In particular, a logistic-regression meta-fusion classifier is **not implemented or fitted** in the saved enhanced decision-fusion run.

## 3A. Main Pipeline File Reference

### Main Pipeline File Reference

| Pipeline stage | Exact relative script/path | Main function or class | Input received | Processing performed | Output produced | Consumed by | Execution mode |
|---|---|---|---|---|---|---|---|
| Main entry point | **None exists for the whole repository** | No repository-wide `main.py` or shell orchestrator | N/A | Executes stages manually in separate scripts | N/A | N/A | No all-in-one entry point |
| Configuration | `FUSION/config.py` | Module constants | Workspace location and source values | Defines subject list, dimensions, epochs, learning rates, dropout, seed, and result directories | Imported constants and directories | Dataset and all training scripts | Imported; also creates directories on import |
| Dataset loading | `FUSION/dataset.py` | `load_subject(subject_id)` | `<subject>_nback_preprocessed.mat` | Searches HDF5/MAT fields, standardizes EEG to `(27,28,8400)` and fNIRS to `(27,36,420,2)`, converts labels to binary | Dict containing EEG, fNIRS, original/binary labels, subject and trial IDs | Verification, training and fusion scripts | Imported library; no `main()` |
| Shared utilities | `FUSION/utils.py` | `set_seed()`, `calculate_channel_statistics()`, `normalize_with_statistics()`, `calculate_class_weights()`, `convert_to_binary_labels()` | Raw arrays and training fold labels | Seeds, channel-wise fold-safe normalization, inverse-frequency class weights | Normalized arrays, weight tensor | Training and dataset scripts | Imported library |
| Raw epoch extraction | `FUSION/preprocess/extract_nback_epochs.m` | Sequential MATLAB script | Raw paired EEG/fNIRS dataset | Loads source data and extracts 24 subject epoch MAT files | `FUSION/data/epochs/<subject>_nback_epochs.mat` | `preprocess_paired_epochs.m` | Direct MATLAB execution |
| Raw preprocessing | `FUSION/preprocess/preprocess_paired_epochs.m` | Sequential MATLAB script | `FUSION/data/epochs/<subject>_nback_epochs.mat` | Filtering, cleaning, segmentation/alignment preparation and signal preparation | `FUSION/data/epochs_preprocessed/<subject>_nback_preprocessed.mat` | `dataset.py` and all Python training scripts | Direct MATLAB execution |
| Dataset diagnostic | `FUSION/preprocess/check_dataset.py` | `main()` | Epoch files matching `*_nback_epochs.mat` | Checks subject data availability and epoch structure | Console summary | Operator | Executable, but contains a Windows-specific hardcoded epoch path |
| Alignment diagnostics | `FUSION/preprocess/audit_alignment.py` | Top-level code | `FUSION/data/epochs/*.mat` | Audits alignment | `FUSION/results/alignment_audit/alignment_audit.csv` | Operator | Direct Python execution; executes at import/top level |
| Temporal alignment diagnostic | `FUSION/preprocess/audit_temporal_alignment.py` | Top-level code | `FUSION/data/epochs/*.mat` | Audits temporal alignment | `FUSION/results/alignment_audit/temporal_alignment_audit.csv` | Operator | Direct Python execution |
| Preprocessed-data verification | `FUSION/scripts/01_verify_data.py` | `main()` | The 24 MAT files loaded through `dataset.load_subject()` | Verifies shapes, finite values and three original class counts | Console pass/fail; exit code `1` on failure | Before LOSO training | Direct executable |
| Binary-label verification | `FUSION/scripts/02_prepare_dataset.py` | `main()` | Preprocessed subjects and labels | Verifies `0→0`, `{1,2}→1`, gives 216 LOW/432 HIGH totals | Console pass/fail | Before LOSO training | Direct executable |
| LOSO split creation | `FUSION/scripts/03_create_loso_splits.py` | `main()` | 24 configured subjects | Creates cyclic 22/1/1 subject splits | `FUSION/results/loso_binary_cnn_bilstm/splits/fold_01.json` through `fold_24.json` | All LOSO training and evaluation scripts | Direct executable |
| Architecture sanity test | `FUSION/scripts/04_test_model_shapes.py` | `main()` | Dummy EEG and fNIRS tensors | Instantiates/checks EEG, fNIRS and representation-fusion models | Console shape report | Operator | Direct executable |
| EEG architecture | `FUSION/models/eeg_cnn_bilstm.py` | `EEGCNNBiLSTM` | `(B,28,8400)` | Temporal CNN + BiLSTM + temporal mean/max pooling + dropout/classifier | 128-D features and 2-class logits | EEG and representation-fusion training | Imported |
| fNIRS architecture | `FUSION/models/fnirs_cnn.py` | `FNIRSCNN` | `(B,36,420,2)` | Combines two chromophores into 72 temporal channels, CNN and 128-D projection | 128-D features and 2-class logits | fNIRS and representation-fusion training | Imported |
| Representation fusion architecture | `FUSION/models/fusion_model.py` | `MultimodalFusionModel` | EEG/fNIRS encoders and paired tensors | Projects both representations to 128-D, concatenates to 256-D, applies dropout classifier | Fused features and 2-class logits | Fusion LOSO training | Imported |
| Final LOSO architecture | `FUSION/models/final_loso_model.py` | `FinalLOSOMultimodalModel` | `(B,28,8400)` and `(B,36,420,2)` | EEG CNN-BiLSTM attention encoder + fNIRS CNN + concatenation/fusion classifier | Fused features and logits | Final LOSO training/evaluation | Imported |
| EEG training | `FUSION/scripts/05_train_eeg_loso.py` | `main()` plus `train_one_epoch()` / `evaluate_model()` | Fold split JSON, subject MAT files | Fold-wise train-only normalization, class weights, AdamW, best validation-Macro-F1 checkpoint, test inference | `checkpoints/eeg_fold_XX_best.pt`, `predictions/eeg_fold_XX.csv`, `metrics/eeg_loso_results.csv` | `08_evaluate_results.py`, `09_generate_confusion_matrices.py`, fusion scripts | Direct executable |
| fNIRS training | `FUSION/scripts/06_train_fnirs_loso.py` | `main()` plus `train_one_epoch()` / `evaluate_model()` | Fold split JSON, subject MAT files | Same fold-wise LOSO procedure for fNIRS | `checkpoints/fnirs_fold_XX_best.pt`, `predictions/fnirs_fold_XX.csv`, `metrics/fnirs_loso_results.csv` | `08_evaluate_results.py`, `09_generate_confusion_matrices.py`, fusion scripts | Direct executable |
| Representation-fusion training | `FUSION/scripts/07_train_fusion_loso.py` | `main()` | EEG/fNIRS fold data plus existing unimodal checkpoints | Loads pretrained encoders, train-only normalization, freezes/then fuses representations, selects best validation-Macro-F1 checkpoint | `checkpoints/fusion_fold_XX_best.pt`, `predictions/fusion_fold_XX.csv`, `metrics/fusion_loso_results.csv` | `08_evaluate_results.py`, `09_generate_confusion_matrices.py` | Direct executable; requires EEG/fNIRS checkpoints |
| Metric aggregation | `FUSION/scripts/08_evaluate_results.py` | `main()` and `load_model_predictions()` | Per-fold EEG/fNIRS/fusion prediction CSV files | Loads 648 predictions, calculates accuracy, precision, recall, Macro-F1 and ROC-AUC, produces summary | `metrics/*_fold_metrics.csv`, `summaries/model_comparison_summary.csv` | `09_generate_confusion_matrices.py`, `10_generate_final_report.py` | Direct executable |
| Confusion matrices | `FUSION/scripts/09_generate_confusion_matrices.py` | `main()` and `generate_matrix()` | EEG/fNIRS/fusion prediction CSV files | Pooled confusion matrices and class recall | `confusion_matrices/*.png/.csv`, `summaries/class_recall_summary.csv` | Operator/report scripts | Direct executable |
| Baseline report generation | `FUSION/scripts/10_generate_final_report.py` | `main()` | `summaries/model_comparison_summary.csv` and pooled classification report | Creates comparison table, performance plot and text report | `summaries/final_model_comparison.csv`, `summaries/model_comparison_plot.png`, `summaries/FINAL_REPORT.txt` | Operator | Direct executable |
| Decision fusion implementation | `FUSION/models/decision_fusion.py` | `average_probability_fusion()`, `weighted_probability_fusion()`, `probabilities_to_predictions()` | Two `(N,2)` probability tensors | Equal or validation-selected weighted averaging and argmax | Fused probability tensor and prediction tensor | Decision-fusion scripts | Imported |
| Decision-fusion training/inference | `FUSION/scripts/decision_level_fusion/18_train_late_fusion.py` | `main()` | 24 LOSO modality checkpoints and held-out test data | Loads trained EEG/fNIRS checkpoints, normalizes test data, averages probabilities at 50:50 | `results/decision_level_fusion/predictions/*_late_fusion_predictions.csv`, fold metrics | Decision-fusion evaluation/report scripts | Direct executable |
| Decision-fusion evaluation | `FUSION/scripts/decision_level_fusion/19_evaluate_late_fusion.py` | Top-level script | Late-fusion prediction CSV files | Calculates accuracy, Macro-F1, ROC-AUC and summaries | `results/decision_level_fusion/metrics/` | Report/confusion scripts | Direct executable at top level |
| Decision-fusion confusion | `FUSION/scripts/decision_level_fusion/20_generate_confusion_matrix.py` | Top-level script | Late-fusion prediction CSV files | Creates late-fusion confusion matrix | `results/decision_level_fusion/confusion_matrices/late_fusion_confusion_matrix.csv` and plot | Operator | Direct executable at top level |
| Decision-fusion report | `FUSION/scripts/decision_level_fusion/21_generate_final_report.py` | Top-level script | Decision-fusion metrics and baseline comparisons | Creates model-comparison report | `results/decision_level_fusion/FINAL_REPORT.md` or text/CSV outputs | Operator | Direct executable at top level |
| Enhanced decision-fusion runner | `FUSION/results/new_result/decision_fusion_enhanced_20261013_134000/run_experiment.py` | `main()`, `fit_temperature()`, `conservative_threshold()` | Existing saved LOSO checkpoints and 50:50 prediction files | Uses per-fold validation temperature scaling and guarded threshold grid; no retraining and no logistic classifier | Timestamped/this run directory predictions, candidate CSV files and `final_report.md` | Saved experimental review | Standalone executable, but is a saved artifact runner |

### Final LOSO pipeline files

| Stage | Exact relative path | Responsibility |
|---|---|---|
| Tune/train | `FUSION/scripts/final_loso/01_tune_dropout_loso.py` | 24-fold LOSO training, dropout `0.3`, max 12 epochs, AdamW, class-weighted cross-entropy, validation-Macro-F1 early stopping and `ReduceLROnPlateau(mode=max,factor=0.5,patience=2,min_lr=1e-6)` |
| Checkpoints | `FUSION/results/final_loso/checkpoints/fold_XX_best.pt` | Saved final model state, dropout, best epoch, validation F1, seed and scheduler metadata |
| Testing | `FUSION/scripts/final_loso/02_evaluate_final_loso.py` | Loads each fold checkpoint and saves `.npz` held-out predictions |
| Confusion matrices | `FUSION/scripts/final_loso/03_generate_confusion_matrix.py` | Pooled fold predictions and final LOSO confusion matrices/plots |
| Final report | `FUSION/scripts/final_loso/04_generate_final_report.py` | Loads `.npz` outputs/checkpoints and writes final text/CSV metrics |

The final LOSO script is guarded by `if __name__ == "__main__"` and is executable directly. Its evaluation and report scripts use relative result paths and are safest to run from the `FUSION/` project directory.

## 3B. Reconstruct the Exact Execution Order

### A. Primary 24-fold baseline LOSO pipeline

| Step | Exact script | Independent? | Required input | Output | Next step dependency |
|---|---|---|---|---|---|
| 1 | MATLAB `FUSION/preprocess/extract_nback_epochs.m` | Yes, if raw data has not already been extracted | Raw EEG/fNIRS dataset at hardcoded MATLAB source path | `FUSION/data/epochs/<subject>_nback_epochs.mat` | Required for preprocessing |
| 2 | MATLAB `FUSION/preprocess/preprocess_paired_epochs.m` | Yes, if processed MAT files do not exist | Extracted epoch MAT files | `FUSION/data/epochs_preprocessed/<subject>_nback_preprocessed.mat` | Required by `dataset.py` |
| 3 | `FUSION/scripts/01_verify_data.py` | Yes | 24 processed subject MAT files | Console verification | Stop if it fails |
| 4 | `FUSION/scripts/02_prepare_dataset.py` | Yes | 24 processed subject MAT files | Console binary-label verification | Stop if it fails |
| 5 | `FUSION/scripts/03_create_loso_splits.py` | Yes | `config.PAIRED_SUBJECTS` | `results/loso_binary_cnn_bilstm/splits/fold_XX.json` | Required by all LOSO training |
| 6 | `FUSION/scripts/04_test_model_shapes.py` | Optional sanity check | None; constructs dummy tensors | Console shape verification | Recommended before training |
| 7 | `FUSION/scripts/05_train_eeg_loso.py` | Yes | 24 processed MAT files and 24 split JSON files | EEG fold checkpoints and prediction CSV files | Required by fusion and decision scripts |
| 8 | `FUSION/scripts/06_train_fnirs_loso.py` | Yes | 24 processed MAT files and 24 split JSON files | fNIRS fold checkpoints and prediction CSV files | Required by fusion and decision scripts |
| 9 | `FUSION/scripts/07_train_fusion_loso.py` | Yes | EEG/fNIRS checkpoints, processed MAT files, splits | Representation-fusion checkpoints and prediction CSV files | Required for fusion evaluation |
| 10 | `FUSION/scripts/08_evaluate_results.py` | Yes | EEG/fNIRS/fusion prediction CSV files | Fold metrics and model comparison CSV | Required for confusion/report steps |
| 11 | `FUSION/scripts/09_generate_confusion_matrices.py` | Yes | EEG/fNIRS/fusion prediction CSV files | Raw and normalized confusion matrices | Can run after step 10 |
| 12 | `FUSION/scripts/10_generate_final_report.py` | Yes | Model comparison and pooled classification CSV files | Final table, plot and text report | Final primary-pipeline report |

If preprocessing or data loading is changed, rerun verification, splits, and every LOSO training script. If only a report/metric script is changed, rerun that script and downstream reporting only.

### B. Final tuned LOSO pipeline

| Step | Exact script | Required input | Output | Next dependency |
|---|---|---|---|---|
| 1 | `FUSION/scripts/03_create_loso_splits.py` | Configured 24 subjects | `fold_01.json`–`fold_24.json` | Required |
| 2 | `FUSION/scripts/final_loso/01_tune_dropout_loso.py` | Preprocessed MAT files and split JSONs | `results/final_loso/checkpoints/fold_XX_best.pt` | Required |
| 3 | `FUSION/scripts/final_loso/02_evaluate_final_loso.py` | Final checkpoints and split JSONs | `results/final_loso/predictions/<subject>.npz` | Required |
| 4 | `FUSION/scripts/final_loso/03_generate_confusion_matrix.py` | Final NPZ predictions | `results/final_loso/confusion_matrices/*.csv/.png` and subject metrics | Required only for confusion/report |
| 5 | `FUSION/scripts/final_loso/04_generate_final_report.py` | Final NPZ files and checkpoints | `results/final_loso/metrics/*.csv`, `results/final_loso/summaries/final_model_summary.csv`, `FINAL_LOSO_REPORT.txt` | Final report |

`final_loso/03_generate_confusion_matrix.py` and `04_generate_final_report.py` execute top-level code and should be run from `FUSION/` because some paths are relative to the working directory.

### C. Decision-level/late-fusion pipeline

1. Complete steps 1–8 of the primary LOSO pipeline; the script requires existing EEG and fNIRS fold checkpoints.
2. Run `FUSION/scripts/decision_level_fusion/18_train_late_fusion.py` to load modality checkpoints and save 50:50 probability fusion predictions.
3. Run `FUSION/scripts/decision_level_fusion/19_evaluate_late_fusion.py`.
4. Optionally run `20_generate_confusion_matrix.py` and `21_generate_final_report.py`.

### D. Enhanced decision-fusion experiment

The saved enhanced runner is independent of encoder retraining. It consumes existing 24-fold EEG/fNIRS checkpoints and existing 50:50 saved test probabilities. The runner's source is:

`FUSION/results/new_result/decision_fusion_enhanced_20261013_134000/run_experiment.py`

Execution sequence:

1. Existing split files must be present under `results/loso_binary_cnn_bilstm/splits/`.
2. Existing EEG/fNIRS checkpoints must be present under `results/loso_binary_cnn_bilstm/checkpoints/` (see exact spelling warning below).
3. Existing baseline predictions must be present under `results/decision_level_fusion/predictions/`.
4. Execute the saved runner; it reloads the existing fold checkpoints only for validation inference.
5. It writes all candidate outputs to `FUSION/results/new_result/decision_fusion_enhanced_20261013_134000/`.

The run previously reported `0.6759` mean LOSO accuracy for `calibrated_conservative_threshold`; this was not logistic-regression accuracy. Logistic meta-fusion was skipped.

### E. Alternative 18/3/3 pipeline

| Step | Script |
|---|---|
| 1 | `FUSION/scripts/alternative_split/11_create_subject_split.py` |
| 2 | `FUSION/scripts/alternative_split/12_train_eeg.py` |
| 3 | `FUSION/scripts/alternative_split/13_train_fnirs.py` |
| 4 | `FUSION/scripts/alternative_split/14_train_fusion.py` |
| 5 | `FUSION/scripts/alternative_split/15_evaluate_results.py` |
| 6 | `FUSION/scripts/alternative_split/16_generate_confusion_matrices.py` |
| 7 | `FUSION/scripts/alternative_split/17_generate_final_report.py` |

Scripts 12–17 in this folder do not have separate `main()` guards; they execute at module/top level when run with `python <script>.py`.

## 3C. Exact Terminal Commands

### A. Environment setup

Open **PowerShell** in the parent workspace, `C:\Users\Admin\Downloads\FUSION`, then enter the project root:

```powershell
cd "C:\Users\Admin\Downloads\FUSION\FUSION"
```

Create a new virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository's existing workspace also contains `C:\Users\Admin\Downloads\FUSION\%VENV%\Scripts\Activate.ps1`. A literal environment directory can be activated with:

```powershell
& "C:\Users\Admin\Downloads\FUSION\%VENV%\Scripts\Activate.ps1"
```

Do not switch to the virtual-environment directory as the working project directory; activate it and remain in `C:\Users\Admin\Downloads\FUSION\FUSION`.

Linux/macOS alternative:

```bash
cd /path/to/FUSION/FUSION
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The source code checks for CUDA automatically and otherwise uses CPU:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

No script is known to require CUDA. CPU execution is valid. On CUDA systems, the matching PyTorch CUDA build must already be installed.

### B. Dataset and preprocessing

The raw-data paths are currently **hardcoded inside MATLAB scripts**. They must be edited or available exactly as written before MATLAB preprocessing can run:

- Raw root: `D:\major_project_group50\dataset`
- Extracted epochs: `D:\major_project_group50\project\FUSION\data\epochs`
- Preprocessed epochs: `D:\major_project_group50\project\FUSION\data\epochs_preprocessed`

Run in MATLAB, in this order:

```text
run('C:\Users\Admin\Downloads\FUSION\FUSION\preprocess\extract_nback_epochs.m')
run('C:\Users\Admin\Downloads\FUSION\FUSION\preprocess\preprocess_paired_epochs.m')
```

There is no verified PowerShell/Python command for MATLAB preprocessing because the source is MATLAB. A code change is required if paths differ from the hardcoded `D:\...` paths.

After preprocessing, verify the Python dataset:

```powershell
python .\scripts\01_verify_data.py
python .\scripts\02_prepare_dataset.py
python .\scripts\03_create_loso_splits.py
python .\scripts\04_test_model_shapes.py
```

For each command:

- Step 01 requires `data\epochs_preprocessed\<subject>_nback_preprocessed.mat` and creates no dataset files.
- Step 02 verifies binary labels and creates no files.
- Step 03 creates `results\loso_binary_cnn_bilstm\splits\fold_XX.json`.
- Step 04 creates no files and only verifies model tensors.

Optional diagnostics:

```powershell
python .\preprocess\check_dataset.py
python .\preprocess\audit_alignment.py
python .\preprocess\audit_temporal_alignment.py
```

`check_dataset.py` has its own Windows-specific hardcoded epoch path and is not the normal preprocessed-data verifier.

### C. Baseline model training

Run from `C:\Users\Admin\Downloads\FUSION\FUSION`:

```powershell
python .\scripts\05_train_eeg_loso.py
```

Requires split JSONs and 24 preprocessed subject files. Produces EEG fold checkpoints, per-fold predictions and EEG metrics.

```powershell
python .\scripts\06_train_fnirs_loso.py
```

Requires split JSONs and 24 preprocessed subject files. Produces fNIRS fold checkpoints, per-fold predictions and fNIRS metrics.

```powershell
python .\scripts\07_train_fusion_loso.py
```

Requires all 24 EEG and fNIRS checkpoints plus the split JSONs and preprocessed data. Produces representation-fusion checkpoints, predictions and metrics.

### D. Final tuned LOSO pipeline

```powershell
python .\scripts\03_create_loso_splits.py
python .\scripts\final_loso\01_tune_dropout_loso.py
python .\scripts\final_loso\02_evaluate_final_loso.py
python .\scripts\final_loso\03_generate_confusion_matrix.py
python .\scripts\final_loso\04_generate_final_report.py
```

- Training requires 24 processed subject files and the LOSO split JSONs.
- Training outputs `results\final_loso\checkpoints\fold_XX_best.pt`.
- Evaluation requires those checkpoints and outputs `results\final_loso\predictions\<subject>.npz`.
- Confusion/report scripts generate CSV, PNG and text summaries under `results\final_loso\`.

### E. Decision/late fusion

After baseline EEG and fNIRS LOSO training:

```powershell
python .\scripts\decision_level_fusion\18_train_late_fusion.py
python .\scripts\decision_level_fusion\19_evaluate_late_fusion.py
python .\scripts\decision_level_fusion\20_generate_confusion_matrix.py
python .\scripts\decision_level_fusion\21_generate_final_report.py
```

The first command loads:

- `results\loso_binary_cnn_bilstm\splits\fold_XX.json`
- `results\loso_binary_cnn_bilstm\checkpoints\eeg_fold_XX_best.pt`
- `results\loso_binary_cnn_bilstm\checkpoints\fnirs_fold_XX_best.pt`
- `data\epochs_preprocessed\<subject>_nback_preprocessed.mat`

It produces `results\decision_level_fusion\predictions\` and metric files.

The bounded validation-only decision-fusion experiment can be run with:

```powershell
python .\scripts\decision_level_fusion\22_run_bounded_v2.py --max-runtime-minutes 30
```

The valid range is greater than `0` and at most `30`; the default is `30`.

The saved enhanced runner can be rerun only if its required checkpoints/predictions are present:

```powershell
python .\results\new_result\decision_fusion_enhanced_20261013_134000\run_experiment.py
```

It writes under `FUSION\results\new_result\decision_fusion_enhanced_20261013_134000\`.

### F. Alternative 18/3/3 pipeline

Run from `FUSION/`:

```powershell
python .\scripts\alternative_split\11_create_subject_split.py
python .\scripts\alternative_split\12_train_eeg.py
python .\scripts\alternative_split\13_train_fnirs.py
python .\scripts\alternative_split\14_train_fusion.py
python .\scripts\alternative_split\15_evaluate_results.py
python .\scripts\alternative_split\16_generate_confusion_matrices.py
python .\scripts\alternative_split\17_generate_final_report.py
```

Outputs are primarily under `results\alternative_split_binary\`.

### G. Optional experimental architectures

Run each family in numerical order:

```powershell
python .\scripts\eeg_attention\01_train_eeg_attention_loso.py
python .\scripts\eeg_attention\02_evaluate_eeg_attention.py
python .\scripts\eeg_attention\03_generate_confusion_matrix.py
python .\scripts\eeg_attention\04_generate_final_report.py
```

Equivalent paths exist for `scripts/fnirs_attention/`, `scripts/fnirs_dual/`, and `scripts/cross_modal_attention/`. Gated-fusion scripts use `scripts/gated_fusion/01_train_gated_fusion_loso.py` through `04_generate_final_report.py` and execute in top-level mode where applicable.

### H. Complete primary end-to-end command list

```powershell
cd C:\Users\Admin\Downloads\FUSION\FUSION
& C:\Users\Admin\Downloads\FUSION\%VENV%\Scripts\Activate.ps1
python .\scripts\01_verify_data.py
python .\scripts\02_prepare_dataset.py
python .\scripts\03_create_loso_splits.py
python .\scripts\04_test_model_shapes.py
python .\scripts\05_train_eeg_loso.py
python .\scripts\06_train_fnirs_loso.py
python .\scripts\07_train_fusion_loso.py
python .\scripts\08_evaluate_results.py
python .\scripts\09_generate_confusion_matrices.py
python .\scripts\10_generate_final_report.py
```

No preprocessing MATLAB command is included in this consolidated Python list because raw data is not processed by Python in this repository.

## 3D. Input-to-Output File Path Mapping

| Pipeline stage | Exact script | Actual input | Actual output/naming pattern | Consumed by |
|---|---|---|---|---|
| Raw extraction | `preprocess/extract_nback_epochs.m` | Hardcoded MATLAB raw dataset | `data/epochs/<subject>_nback_epochs.mat` | Preprocessing MATLAB script |
| Preprocessing | `preprocess/preprocess_paired_epochs.m` | `data/epochs/<subject>_nback_epochs.mat` | `data/epochs_preprocessed/<subject>_nback_preprocessed.mat` | `dataset.py` and every Python LOSO script |
| Dataset loading | `dataset.py` → `load_subject()` | `data/epochs_preprocessed/<subject>_nback_preprocessed.mat` | In-memory `(27,28,8400)` EEG, `(27,36,420,2)` fNIRS, labels | Verification and training loops |
| LOSO split | `scripts/03_create_loso_splits.py` | `config.PAIRED_SUBJECTS` | `results/loso_binary_cnn_bilstm/splits/fold_XX.json` | EEG/fNIRS/fusion/final LOSO and enhanced fusion |
| EEG checkpoint | `scripts/05_train_eeg_loso.py` | Fold data and split JSON | `results/loso_binary_cnn_bilstm/checkpoints/eeg_fold_XX_best.pt` | fNIRS-independent evaluation, fusion, late fusion |
| fNIRS checkpoint | `scripts/06_train_fnirs_loso.py` | Fold data and split JSON | `results/loso_binary_cnn_bilstm/checkpoints/fnirs_fold_XX_best.pt` | Fusion and late fusion |
| EEG prediction | `scripts/05_train_eeg_loso.py` | Held-out test subject | `results/loso_binary_cnn_bilstm/predictions/eeg_fold_XX.csv` | `08_evaluate_results.py`, confusion/report scripts |
| fNIRS prediction | `scripts/06_train_fnirs_loso.py` | Held-out test subject | `results/loso_binary_cnn_bilstm/predictions/fnirs_fold_XX.csv` | `08_evaluate_results.py`, confusion/report scripts |
| Representation-fusion checkpoint | `scripts/07_train_fusion_loso.py` | Pretrained EEG/fNIRS encoders and fold data | `results/loso_binary_cnn_bilstm/checkpoints/fusion_fold_XX_best.pt` | Representation-fusion evaluation |
| Representation-fusion prediction | `scripts/07_train_fusion_loso.py` | Held-out test subject | `results/loso_binary_cnn_bilstm/predictions/fusion_fold_XX.csv` | `08_evaluate_results.py` |
| Fold metrics | `scripts/05`–`07` | Held-out predictions | `results/loso_binary_cnn_bilstm/metrics/eeg_loso_results.csv`, `fnirs_loso_results.csv`, `fusion_loso_results.csv` | Evaluation aggregation |
| Aggregated metrics | `scripts/08_evaluate_results.py` | Three per-fold prediction directories | `metrics/<model>_fold_metrics.csv`, `summaries/model_comparison_summary.csv` | Confusion/report scripts |
| Confusion matrices | `scripts/09_generate_confusion_matrices.py` | Three prediction CSV sets | `confusion_matrices/*.png/.csv`, `summaries/class_recall_summary.csv` | Operator |
| Primary report | `scripts/10_generate_final_report.py` | Model comparison CSV | `summaries/final_model_comparison.csv`, `summaries/model_comparison_plot.png`, `summaries/FINAL_REPORT.txt` | Operator |
| Final LOSO checkpoint | `scripts/final_loso/01_tune_dropout_loso.py` | Fold training/validation data | `results/final_loso/checkpoints/fold_XX_best.pt` | Final LOSO evaluator |
| Final LOSO prediction | `scripts/final_loso/02_evaluate_final_loso.py` | Final checkpoint and held-out subject | `results/final_loso/predictions/<subject>.npz` | Final confusion/report scripts |
| Final LOSO metrics | `scripts/final_loso/04_generate_final_report.py` | NPZ predictions and checkpoint metadata | `results/final_loso/metrics/fold_metrics.csv`, `subject_wise_results.csv`, `pooled_*`, `summaries/final_model_summary.csv`, `summaries/FINAL_LOSO_REPORT.txt` | Operator |
| Late-fusion prediction | `scripts/decision_level_fusion/18_train_late_fusion.py` | Existing EEG/fNIRS probabilities/checkpoints | `results/decision_level_fusion/predictions/<subject>_late_fusion_predictions.csv` | Decision-fusion evaluator/confusion/report |
| Enhanced candidate prediction | `results/new_result/decision_fusion_enhanced_20261013_134000/run_experiment.py` | Existing fold probabilities/checkpoints | `predictions/baseline/heldout_predictions.csv`, `predictions/candidates/heldout_predictions.csv`, `candidate_fold_metrics.csv`, `candidate_summary.csv`, `selected_parameters.csv`, `final_report.md` | Operator |

The repository does not implement a separate Micro-F1 metric in the primary scripts. Macro-F1 is explicitly used for validation and final reporting. Balanced accuracy is implemented in decision-fusion experimental scripts, such as `22_run_bounded_v2.py`, and in the enhanced saved runner.

## 3E. Troubleshooting and Execution Dependencies

### Confirmed issues

1. **No one-command project entry point**: no root `main.py` or orchestration shell script exists. Run the stage scripts in the documented sequence.
2. **Missing packages**: install `requirements.txt` after activating the environment. The repository uses NumPy, SciPy, h5py, pandas, scikit-learn, matplotlib, seaborn and PyTorch.
3. **Missing MAT files**: verify `FUSION/data/epochs_preprocessed/` contains all 24 `<subject>_nback_preprocessed.mat` files. `01_verify_data.py` gives the exact first failure.
4. **Hardcoded MATLAB paths**: `extract_nback_epochs.m` and `preprocess_paired_epochs.m` use `D:\major_project_group50\...`. These paths must point to your raw and project directories; a code modification is required for a different absolute location.
5. **Hardcoded Windows preprocessing check**: `preprocess/check_dataset.py` uses `D:\major_project_group50\project\FUSION\data\epochs` and is not portable as written.
6. **Working directory**: launch Python scripts from `FUSION/`. The scripts derive `PROJECT_ROOT` from `__file__`, but some final/evaluation scripts use relative `results/...` paths.
7. **Missing LOSO splits**: run `03_create_loso_splits.py` before any LOSO training.
8. **Missing modality checkpoints**: representation fusion needs EEG and fNIRS checkpoints. Late fusion also needs both checkpoint sets.
9. **GPU is optional**: scripts automatically select CUDA if available and otherwise CPU. CPU execution is supported; CUDA is not required.
10. **Training scripts overwrite earlier checkpoints**: rerunning EEG/fNIRS/fusion training can replace existing `*_best.pt` and prediction CSV files with the same naming pattern.
11. **Fold checkpoint mapping**: enhanced and decision-fusion scripts verify fold/test-subject and checkpoint-fold IDs; do not manually rename checkpoint files without updating the corresponding split and mapping logic.
12. **No OOF logistic-regression data**: regularized logistic meta-fusion is skipped in the bounded and enhanced decision-fusion runners because independent nested out-of-fold probabilities are not saved. It is not resolved by rerunning the saved scripts.
13. **No CLI for baseline scripts**: `05`, `06`, `07`, `08`, `09`, `10` and final LOSO scripts do not accept custom dataset arguments. Paths come from `config.py` or hardcoded result constants.
14. **One argparse script**: `22_run_bounded_v2.py` accepts `--max-runtime-minutes`. Its valid range is `>0` and `<=30`.
15. **No saved checkpoint means no inference**: testing/evaluation scripts will raise a missing-file or load error. Train or restore the required checkpoint first.
16. **MATLAB/HDF5 format**: `dataset.py` uses h5py and searches recursively for dataset names; MATLAB files must save datasets in a discoverable HDF5 layout.

Resolution that requires source modification should be limited to changing hardcoded MATLAB paths or relative result paths. Before editing, verify whether a more suitable existing result/checkpoint directory already contains the required data.

## 3F. Final Quick-Reference Execution Guide

### Quick Start: Run the Project

1. **Activate environment**

```powershell
cd C:\Users\Admin\Downloads\FUSION\FUSION
& C:\Users\Admin\Downloads\FUSION\%VENV%\Scripts\Activate.ps1
```

2. **Install dependencies**

```powershell
python -m pip install -r requirements.txt
```

3. **Configure dataset:** place/verify all 24 files under `C:\Users\Admin\Downloads\FUSION\FUSION\data\epochs_preprocessed\` with names `<subject>_nback_preprocessed.mat`. MATLAB raw preprocessing is required if these files do not exist; its source paths currently require verification.

4. **Preprocess/check data**

```powershell
python .\scripts\01_verify_data.py
python .\scripts\02_prepare_dataset.py
python .\scripts\03_create_loso_splits.py
```

5. **Train EEG**

```powershell
python .\scripts\05_train_eeg_loso.py
```

6. **Train fNIRS**

```powershell
python .\scripts\06_train_fnirs_loso.py
```

7. **Run representation fusion**

```powershell
python .\scripts\07_train_fusion_loso.py
```

8. **Run validation/test generation/evaluation**

```powershell
python .\scripts\08_evaluate_results.py
python .\scripts\09_generate_confusion_matrices.py
python .\scripts\10_generate_final_report.py
```

9. **Run final tuned LOSO alternative**

```powershell
python .\scripts\final_loso\01_tune_dropout_loso.py
python .\scripts\final_loso\02_evaluate_final_loso.py
python .\scripts\final_loso\03_generate_confusion_matrix.py
python .\scripts\final_loso\04_generate_final_report.py
```

10. **Run late fusion**

```powershell
python .\scripts\decision_level_fusion\18_train_late_fusion.py
python .\scripts\decision_level_fusion\19_evaluate_late_fusion.py
python .\scripts\decision_level_fusion\20_generate_confusion_matrix.py
python .\scripts\decision_level_fusion\21_generate_final_report.py
```

### Quick Reference Table

| What I want to do | Script to execute | Terminal command | Output location |
|---|---|---|---|
| Verify/preprocess data | `scripts/01_verify_data.py` | `python .\scripts\01_verify_data.py` | Console only |
| Create LOSO splits | `scripts/03_create_loso_splits.py` | `python .\scripts\03_create_loso_splits.py` | `results/loso_binary_cnn_bilstm/splits/fold_XX.json` |
| Train EEG | `scripts/05_train_eeg_loso.py` | `python .\scripts\05_train_eeg_loso.py` | `results/loso_binary_cnn_bilstm/checkpoints/eeg_fold_XX_best.pt` |
| Train fNIRS | `scripts/06_train_fnirs_loso.py` | `python .\scripts\06_train_fnirs_loso.py` | `results/loso_binary_cnn_bilstm/checkpoints/fnirs_fold_XX_best.pt` |
| Train representation fusion | `scripts/07_train_fusion_loso.py` | `python .\scripts\07_train_fusion_loso.py` | `results/loso_binary_cnn_bilstm/checkpoints/fusion_fold_XX_best.pt` |
| Run final tuned LOSO | `scripts/final_loso/01_tune_dropout_loso.py` | `python .\scripts\final_loso\01_tune_dropout_loso.py` | `results/final_loso/checkpoints/fold_XX_best.pt` |
| Run late fusion | `scripts/decision_level_fusion/18_train_late_fusion.py` | `python .\scripts\decision_level_fusion\18_train_late_fusion.py` | `results/decision_level_fusion/predictions/` |
| Generate predictions | EEG/fNIRS/fusion training scripts or final evaluator | See the corresponding training/evaluator command above | Per-model `predictions/` folders |
| Evaluate results | `scripts/08_evaluate_results.py` | `python .\scripts\08_evaluate_results.py` | `results/loso_binary_cnn_bilstm/metrics/` and `summaries/` |
| Generate confusion matrices | `scripts/09_generate_confusion_matrices.py` | `python .\scripts\09_generate_confusion_matrices.py` | `results/loso_binary_cnn_bilstm/confusion_matrices/` |
| Generate final report | `scripts/10_generate_final_report.py` | `python .\scripts\10_generate_final_report.py` | `results/loso_binary_cnn_bilstm/summaries/FINAL_REPORT.txt` |
