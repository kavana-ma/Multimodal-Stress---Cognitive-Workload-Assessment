from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXPERIMENT_STARTED = time.monotonic()
EXPERIMENT_STARTED_AT = datetime.now(timezone.utc)
MAX_RUNTIME_SECONDS = 20 * 60
WORKFLOW_STARTED_AT = datetime.fromisoformat("2026-10-13T13:38:59.071788+05:30")
WORKFLOW_DEADLINE = WORKFLOW_STARTED_AT + timedelta(minutes=20)
SEED = 42

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, PAIRED_SUBJECTS
from dataset import load_subject
from models.decision_fusion import average_probability_fusion
from models.eeg_cnn_bilstm import EEGCNNBiLSTM
from models.fnirs_cnn import FNIRSCNN


BASELINE_ROOT = PROJECT_ROOT / "results" / "decision_level_fusion"
MODEL_RESULT_ROOT = PROJECT_ROOT / "results" / "loso_binary_cnn_bilstm"
PRIOR_V2_ROOT = PROJECT_ROOT / "results" / "decision_level_fusion_v2_20261013_114310"
ALT_ROOT = PROJECT_ROOT / "results" / "alternative_split_binary"
WEIGHTED_V1_ROOT = PROJECT_ROOT / "results" / "decision_level_fusion_weighted_v1"
THRESHOLD_GRID = (0.45, 0.50, 0.55)
TEMPERATURE_GRID = (0.80, 0.90, 1.00, 1.10, 1.25)
MIN_VALIDATION_F1_GAIN = 0.05
METRICS = (
    "accuracy", "macro_f1", "balanced_accuracy", "low_recall", "high_recall",
    "macro_precision", "macro_recall", "roc_auc", "pr_auc", "mcc",
    "brier_score", "log_loss", "ece",
)


class DeadlineReached(RuntimeError):
    pass


def log(message: str) -> None:
    text = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(text, flush=True)
    with (RUN_DIR / "run_log.txt").open("a", encoding="utf-8") as output:
        output.write(text + "\n")


def check_deadline(stage: str) -> None:
    elapsed = time.monotonic() - EXPERIMENT_STARTED
    if elapsed >= MAX_RUNTIME_SECONDS or datetime.now(timezone.utc) >= WORKFLOW_DEADLINE.astimezone(timezone.utc):
        raise DeadlineReached(f"20-minute wall-clock deadline reached during {stage}")


def write_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_train(test_subject: str, validation_subject: str) -> list[str]:
    return [s for s in PAIRED_SUBJECTS if s not in (test_subject, validation_subject)]


def metric_values(y_true: np.ndarray, p_high: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    p_high = np.asarray(p_high, dtype=np.float64)
    if len(y_true) != len(p_high) or len(y_true) == 0:
        raise ValueError("Labels and probabilities must be non-empty and aligned")
    if not np.isfinite(p_high).all() or np.any(p_high < 0) or np.any(p_high > 1):
        raise ValueError("Invalid class-1 probabilities")
    y_pred = (p_high > threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    clipped = np.clip(p_high, 1e-7, 1 - 1e-7)
    y_low = 1 - y_true
    ece = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        mask = (p_high >= lower) & (p_high < upper if upper < 1 else p_high <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(p_high[mask].mean()) - float(y_true[mask].mean()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "low_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "high_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_high)) if np.unique(y_true).size == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, p_high)) if np.unique(y_true).size == 2 else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(np.mean((p_high - y_true) ** 2)),
        "log_loss": float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))),
        "ece": float(ece),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_samples": int(len(y_true)),
        "n_low": int((y_true == 0).sum()),
        "n_high": int((y_true == 1).sum()),
    }


def predict_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp[:, 1] / exp.sum(axis=1)


def fit_temperature(val_logits: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    candidates = []
    for temperature in TEMPERATURE_GRID:
        p = predict_from_logits(val_logits, temperature)
        values = metric_values(labels, p)
        candidates.append((values["log_loss"], abs(temperature - 1.0), temperature, values))
    selected = min(candidates, key=lambda row: (row[0], row[1]))
    return float(selected[2]), selected[3]


def conservative_threshold(labels: np.ndarray, p_high: np.ndarray) -> tuple:
    default = metric_values(labels, p_high, 0.5)
    scored = []
    for threshold in THRESHOLD_GRID:
        values = metric_values(labels, p_high, threshold)
        scored.append((values["macro_f1"], values["balanced_accuracy"], -abs(threshold - 0.5), threshold, values))
    best = max(scored, key=lambda row: row[:4])
    accepted = (
        best[3] != 0.5
        and best[0] >= default["macro_f1"] + MIN_VALIDATION_F1_GAIN
        and best[1] >= default["balanced_accuracy"]
    )
    selected_threshold = float(best[3]) if accepted else 0.5
    selected_metrics = best[4] if accepted else default
    return selected_threshold, selected_metrics, default, float(best[3]), best[4], accepted


def load_validation_inference(modality: str, checkpoint_path: Path, fold: int, subject: str,
                              data: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict]:
    check_deadline(f"fold {fold} {modality} validation inference")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("fold", fold) != fold:
        raise ValueError(f"Checkpoint fold mismatch: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"No model_state_dict in {checkpoint_path}")
    if "train_mean" not in checkpoint or "train_std" not in checkpoint:
        raise KeyError(f"Normalization statistics missing in {checkpoint_path}")

    eeg = data["eeg"]
    fnirs = data["fnirs"]
    if eeg.shape != (27, 28, 8400) or fnirs.shape != (27, 36, 420, 2):
        raise ValueError(f"Unexpected input shapes for {subject}: EEG {eeg.shape}, fNIRS {fnirs.shape}")
    if not np.isfinite(eeg).all() or not np.isfinite(fnirs).all():
        raise ValueError(f"Non-finite input values for {subject}")
    labels = data["labels_binary"].astype(np.int64)
    if len(labels) != 27 or np.bincount(labels, minlength=2).tolist() != [9, 18]:
        raise ValueError(f"Unexpected binary class counts for {subject}")

    if modality == "eeg":
        model = EEGCNNBiLSTM(num_channels=28, num_classes=2, feature_dim=128)
        values = eeg
    else:
        model = FNIRSCNN(num_channels=36, num_chromophores=2, num_classes=2, feature_dim=128)
        values = fnirs
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    mean = np.asarray(checkpoint["train_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["train_std"], dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"Invalid stored normalization values in {checkpoint_path}")
    normalized = ((values.astype(np.float32) - mean) / np.maximum(std, 1e-8)).astype(np.float32)
    logits = []
    with torch.inference_mode():
        for start in range(0, len(labels), 16):
            batch = torch.from_numpy(normalized[start:start + 16]).to(device)
            result = model(batch)
            if "logits" not in result or result["logits"].shape != (len(batch), 2):
                raise ValueError(f"Invalid {modality} logits shape from {checkpoint_path}")
            logits.append(result["logits"].cpu().numpy())
    logits = np.concatenate(logits, axis=0)
    if not np.isfinite(logits).all():
        raise ValueError(f"Non-finite logits from {checkpoint_path}")
    audit = {
        "subject": subject,
        "eeg_shape": list(eeg.shape),
        "fnirs_shape": list(fnirs.shape),
        "labels_original": np.asarray(data["labels_original"]).astype(int).tolist(),
        "labels_binary": labels.tolist(),
        "subject_ids": [str(value) for value in np.asarray(data["subject_id"]).tolist()],
        "trial_ids": np.asarray(data["trial_id"]).astype(int).tolist(),
        "checkpoint_fold": int(checkpoint.get("fold", fold)),
        "checkpoint_path": str(checkpoint_path),
        "normalization_mean_shape": list(mean.shape),
        "normalization_std_shape": list(std.shape),
        "n_samples": len(labels),
        "class_counts": {"LOW": 9, "HIGH": 18},
    }
    del model, normalized, eeg, fnirs
    return logits, labels, audit


def read_saved_test(subject: str, expected_trial_ids: list[int] | None = None) -> tuple[pd.DataFrame, dict]:
    path = BASELINE_ROOT / "predictions" / f"{subject}_late_fusion_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing held-out prediction file: {path}")
    frame = pd.read_csv(path)
    required = {
        "subject_id", "trial_id", "true_label", "eeg_probability_0", "eeg_probability_1",
        "eeg_prediction", "fnirs_probability_0", "fnirs_probability_1", "fnirs_prediction",
        "fused_probability_0", "fused_probability_1", "fused_prediction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(frame) != 27 or frame["subject_id"].nunique() != 1 or str(frame["subject_id"].iloc[0]) != subject:
        raise ValueError(f"Wrong subject/sample count in {path}")
    if frame["trial_id"].duplicated().any():
        raise ValueError(f"Duplicate trial IDs in {path}")
    if expected_trial_ids is not None and frame["trial_id"].astype(int).tolist() != expected_trial_ids:
        raise ValueError(f"Trial alignment failure in {path}")
    probability_columns = [
        "eeg_probability_0", "eeg_probability_1", "fnirs_probability_0", "fnirs_probability_1",
        "fused_probability_0", "fused_probability_1",
    ]
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(f"Invalid probability in {path}")
    eeg = frame[["eeg_probability_0", "eeg_probability_1"]].to_numpy(dtype=np.float64)
    fnirs = frame[["fnirs_probability_0", "fnirs_probability_1"]].to_numpy(dtype=np.float64)
    fused = frame[["fused_probability_0", "fused_probability_1"]].to_numpy(dtype=np.float64)
    if not np.allclose(eeg.sum(1), 1, atol=2e-5) or not np.allclose(fnirs.sum(1), 1, atol=2e-5):
        raise ValueError(f"Modality probabilities do not sum to one in {path}")
    if not np.allclose(fused, (eeg + fnirs) / 2, atol=2e-6):
        raise ValueError(f"Saved equal fusion differs from 50:50 average in {path}")
    expected_predictions = [
        (eeg[:, 1] > 0.5).astype(int), (fnirs[:, 1] > 0.5).astype(int), (fused[:, 1] > 0.5).astype(int),
    ]
    for column, predicted in zip(("eeg_prediction", "fnirs_prediction", "fused_prediction"), expected_predictions):
        if not np.array_equal(frame[column].to_numpy(dtype=int), predicted):
            raise ValueError(f"Saved {column} disagrees with probability argmax in {path}")
    return frame, {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def aggregate(rows: list[dict], group_column: str) -> list[dict]:
    output = []
    for name, group in pd.DataFrame(rows).groupby(group_column):
        result = {group_column: name, "n_folds": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            result[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(result)
    return output


def paired_comparisons(candidate_rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    base = pd.DataFrame(baseline_rows).set_index("fold")
    candidate = pd.DataFrame(candidate_rows)
    rng = np.random.default_rng(SEED)
    output = []
    for method, group in candidate.groupby("method"):
        current = group.set_index("fold")
        row = {"method": method, "n_folds": len(current)}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "low_recall", "high_recall", "roc_auc", "pr_auc", "mcc", "brier_score", "log_loss", "ece"):
            difference = current[metric].to_numpy(dtype=float) - base[metric].to_numpy(dtype=float)
            row[f"{metric}_mean_delta"] = float(np.nanmean(difference))
            row[f"{metric}_improved_folds"] = int((difference > 1e-12).sum()) if metric not in ("brier_score", "log_loss", "ece") else int((difference < -1e-12).sum())
            row[f"{metric}_tied_folds"] = int((np.abs(difference) <= 1e-12).sum())
            row[f"{metric}_worsened_folds"] = int((difference < -1e-12).sum()) if metric not in ("brier_score", "log_loss", "ece") else int((difference > 1e-12).sum())
            if metric == "macro_f1":
                samples = difference[rng.integers(0, len(difference), size=(10000, len(difference)))].mean(axis=1)
                row["macro_f1_delta_ci_95_low"] = float(np.quantile(samples, 0.025))
                row["macro_f1_delta_ci_95_high"] = float(np.quantile(samples, 0.975))
        output.append(row)
    return output


def save_progress(baseline_rows: list[dict], candidate_rows: list[dict], diagnostic_rows: list[dict],
                  selection_rows: list[dict], baseline_predictions: list[dict], candidate_predictions: list[dict]) -> None:
    write_csv(baseline_rows, RUN_DIR / "baseline_metrics.csv")
    write_csv(candidate_rows, RUN_DIR / "candidate_fold_metrics.csv")
    write_csv(diagnostic_rows, RUN_DIR / "prediction_diagnostics.csv")
    write_csv(selection_rows, RUN_DIR / "selected_parameters.csv")
    write_csv(baseline_predictions, RUN_DIR / "predictions" / "baseline" / "heldout_predictions.csv")
    write_csv(candidate_predictions, RUN_DIR / "predictions" / "candidates" / "heldout_predictions.csv")


def capture_source_hashes() -> dict:
    source_paths = [
        PROJECT_ROOT / "config.py", PROJECT_ROOT / "dataset.py", PROJECT_ROOT / "utils.py",
        PROJECT_ROOT / "models" / "eeg_cnn_bilstm.py", PROJECT_ROOT / "models" / "fnirs_cnn.py",
        PROJECT_ROOT / "models" / "decision_fusion.py",
        PROJECT_ROOT / "scripts" / "decision_level_fusion" / "18_train_late_fusion.py",
        PROJECT_ROOT / "scripts" / "decision_level_fusion" / "19_evaluate_late_fusion.py",
        Path(__file__).resolve(),
    ]
    hashes = []
    for path in source_paths:
        hashes.append({"path": str(path), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None})
    optional_weight_script = PROJECT_ROOT / "scripts" / "decision_level_fusion" / "22_train_validation_weighted_fusion.py"
    hashes.append({"path": str(optional_weight_script), "exists": optional_weight_script.is_file(), "sha256": sha256(optional_weight_script) if optional_weight_script.is_file() else None})
    return {"git_commit": None, "git_status": "unavailable: workspace is not a Git repository", "files": hashes}


def make_summary(candidate_rows: list[dict]) -> list[dict]:
    return aggregate(candidate_rows, "method")


def make_report(status: str, audit: dict, environment: dict, baseline_summary: list[dict],
                summary: list[dict], comparisons: list[dict], findings: list[dict],
                runtime: float, feature_runtime: float) -> str:
    table = pd.DataFrame(summary).set_index("method") if summary else pd.DataFrame()
    lines = [
        "# EEG-fNIRS Decision-Fusion Enhanced Experiment",
        "",
        f"**Status:** {status}",
        f"**Primary protocol:** 24-fold subject-independent LOSO (22 train / 1 validation / 1 test per fold).",
        f"**Runtime:** {runtime:.1f} s total; data loading and validation inference: {feature_runtime:.1f} s (20-minute limit: {MAX_RUNTIME_SECONDS} s).",
        f"**Environment:** Python {environment['python']}; Torch {environment['torch']}; device {environment['device']}; temporary environment only.",
        "",
        "## Executive Summary",
        "",
        "The saved 50:50 probability-fusion baseline was verified from all 24 held-out prediction files. Validation-only temperature scaling and a conservative threshold rule were evaluated with the same 648 test trials. Treat these candidate comparisons as exploratory: the same 24 folds informed the investigation, so no independent confirmatory test set was available.",
        "",
        "## Results",
        "",
        "All figures below are fold mean +/- sample standard deviation. Brier, log-loss, and ECE are lower-is-better.",
        "",
        "| Method | Accuracy | Macro-F1 | Balanced acc. | LOW recall | HIGH recall | ROC-AUC | PR-AUC | MCC | Brier | Log loss | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["baseline_50_50", "calibrated_equal_50_50", "conservative_threshold", "calibrated_conservative_threshold", "majority_high"]
    for method in order:
        if method not in table.index:
            continue
        row = table.loc[method]
        lines.append(
            f"| {method} | {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f} "
            f"| {row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f} "
            f"| {row['balanced_accuracy_mean']:.4f} +/- {row['balanced_accuracy_std']:.4f} "
            f"| {row['low_recall_mean']:.4f} +/- {row['low_recall_std']:.4f} "
            f"| {row['high_recall_mean']:.4f} +/- {row['high_recall_std']:.4f} "
            f"| {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f} "
            f"| {row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f} "
            f"| {row['mcc_mean']:.4f} +/- {row['mcc_std']:.4f} "
            f"| {row['brier_score_mean']:.4f} +/- {row['brier_score_std']:.4f} "
            f"| {row['log_loss_mean']:.4f} +/- {row['log_loss_std']:.4f} "
            f"| {row['ece_mean']:.4f} +/- {row['ece_std']:.4f} |"
        )
    lines.extend(["", "### Paired Fold Comparison", "", "Differences are paired by held-out subject. The 95% percentile bootstrap interval resamples folds; LOSO folds share overlapping training cohorts, so this interval is descriptive and does not establish statistical significance.", ""])
    comp = {row["method"]: row for row in comparisons}
    for method in ("calibrated_equal_50_50", "conservative_threshold", "calibrated_conservative_threshold"):
        if method not in comp:
            continue
        row = comp[method]
        lines.append(
            f"- `{method}` vs 50:50: Macro-F1 delta {row['macro_f1_mean_delta']:+.4f} "
            f"(95% bootstrap interval {row['macro_f1_delta_ci_95_low']:+.4f} to {row['macro_f1_delta_ci_95_high']:+.4f}); "
            f"folds improved/tied/worsened {row['macro_f1_improved_folds']}/{row['macro_f1_tied_folds']}/{row['macro_f1_worsened_folds']}."
        )
    baseline_table = pd.DataFrame(baseline_summary).set_index("method") if baseline_summary else pd.DataFrame()
    lines.extend(["", "### Unimodal Controls", "", "| Model | Accuracy | Macro-F1 | Balanced accuracy | LOW recall | HIGH recall | ROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|"])
    for method in ("EEG", "fNIRS", "Fusion"):
        if method not in baseline_table.index:
            continue
        row = baseline_table.loc[method]
        lines.append(
            f"| {method} | {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f} "
            f"| {row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f} "
            f"| {row['balanced_accuracy_mean']:.4f} +/- {row['balanced_accuracy_std']:.4f} "
            f"| {row['low_recall_mean']:.4f} +/- {row['low_recall_std']:.4f} "
            f"| {row['high_recall_mean']:.4f} +/- {row['high_recall_std']:.4f} "
            f"| {row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f} |"
        )
    lines.extend(["", "## Baseline and Cohort Audit", "", "- 24 paired subjects; each has 27 aligned trials, EEG (28 x 8400), fNIRS (36 x 420 x 2). Binary labels are LOW=0 and HIGH=1, with 9/18 trials per subject.", "- The original 50:50 model probabilities are the positive-class softmax probability from each modality; the saved fusion is their arithmetic average. Predictions use the original argmax rule.", "- The active 50:50 summary is checked against cached predictions; the weighted-v1 directory and its proposed training script do not exist at their expected paths.", "- The original 18/3/3 experiment is a separate protocol. Its saved learned representation-fusion result must not be conflated with decision fusion or LOSO.", "- Source hashes and checkpoint paths/sizes are in `source_hashes.json` and `input_audit.json`; no Git commit was available.", "", "## Separate 18/3/3 Evidence", ""])
    prior_alt_path = PRIOR_V2_ROOT / "18_3_3_comparison.csv"
    if prior_alt_path.is_file():
        prior_alt = pd.read_csv(prior_alt_path)
        lines.append("These saved results are secondary and are not pooled with LOSO.")
        for method in ("decision_equal_50_50", "decision_equal_validation_threshold", "decision_validation_weighted", "existing_representation_fusion", "existing_eeg_only", "existing_fnirs_only"):
            matched = prior_alt[prior_alt["method"] == method]
            if len(matched) == 1:
                row = matched.iloc[0]
                lines.append(f"- `{method}`: accuracy {row['pooled_accuracy']:.4f}, Macro-F1 {row['pooled_macro_f1']:.4f}, balanced accuracy {row['pooled_balanced_accuracy']:.4f}, ROC-AUC {row['pooled_roc_auc']:.4f} (81 trials).")
    else:
        lines.append(f"Saved 18/3/3 comparison unavailable at `{prior_alt_path}`.")
    lines.extend(["", "## Diagnostic Findings", ""])
    for finding in findings:
        lines.append(f"- **{finding['severity']} / {finding['status']}: {finding['issue']}** {finding['evidence']} Fix: {finding['fix']}")
    lines.extend(["", "## Decision and Limitations", "", "The 50:50 fusion remains the preferred control unless a candidate's mean LOSO Macro-F1 improves under independent confirmation. Validation calibration uses one 27-trial held-out validation subject per fold, so temperature estimates are noisy and candidate results are exploratory. Conservative threshold movement requires at least +0.05 validation Macro-F1 with no balanced-accuracy decline; otherwise the default 0.50 threshold is kept.", "", "Regularized logistic meta-fusion was skipped because no independent training-subject out-of-fold modality probabilities were available. No optional architecture was trained: the goal was the no-retraining primary decision-fusion experiment, and training a new subject-independent model was not justified by the available saved features or runtime evidence. Raw preprocessing and original encoders were not rerun.", "", f"The Python environment lacked NumPy, h5py, pandas, scikit-learn, and Torch. A disposable Python 3.12/Torch 2.6 CPU environment was used; the global environment and requirements file were not changed. Total runtime was {runtime:.1f} seconds, and validation data loading/inference took {feature_runtime:.1f} seconds.", "", "See `config.json`, `environment.json`, `input_audit.json`, `source_hashes.json`, `selected_parameters.csv`, `candidate_fold_metrics.csv`, `candidate_summary.csv`, `fold_comparison.csv`, `prediction_diagnostics.csv`, and `diagnostic_findings.csv` for the reproducibility record.", ""])
    return "\n".join(lines)


def main() -> int:
    if (RUN_DIR / "run_manifest.json").exists():
        raise FileExistsError("This run directory already contains a manifest; refusing to overwrite or restart it")
    for subdir in ("predictions/baseline", "predictions/candidates"):
        (RUN_DIR / subdir).mkdir(parents=True, exist_ok=False)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration = {
        "experiment_id": RUN_DIR.name,
        "seed": SEED,
        "protocol": "24-fold LOSO with existing 22/1/1 split JSONs",
        "control": "saved equal-weight 50:50 probabilities, threshold=0.50 argmax",
        "temperature_grid": TEMPERATURE_GRID,
        "threshold_grid": THRESHOLD_GRID,
        "threshold_acceptance_rule": {
            "minimum_validation_macro_f1_gain": MIN_VALIDATION_F1_GAIN,
            "balanced_accuracy_must_not_decrease": True,
            "fallback_threshold": 0.5,
        },
        "calibration": "per-fold, per-modality scalar temperature fitted by validation log loss; same calibrated probabilities combined 50:50",
        "maximum_wall_clock_seconds": MAX_RUNTIME_SECONDS,
        "workflow_started_at": WORKFLOW_STARTED_AT.isoformat(),
        "workflow_deadline": WORKFLOW_DEADLINE.isoformat(),
        "data_directory": str(DATA_DIR),
        "baseline_prediction_directory": str(BASELINE_ROOT / "predictions"),
        "checkpoint_directory": str(MODEL_RESULT_ROOT / "checkpoints"),
        "split_directory": str(MODEL_RESULT_ROOT / "splits"),
        "weighted_v1_directory": str(WEIGHTED_V1_ROOT),
        "weighted_v1_script": str(PROJECT_ROOT / "scripts" / "decision_level_fusion" / "22_train_validation_weighted_fusion.py"),
        "output_directory": str(RUN_DIR),
        "output_directory_checked_absent_before_creation": True,
        "source_root_is_git_repository": False,
    }
    write_json(configuration, RUN_DIR / "config.json")

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "torch_cuda_available": torch.cuda.is_available(),
        "global_environment_modified": False,
        "environment_note": "Dependencies installed into a disposable venv outside the repository; venv removed after run.",
    }
    write_json(environment, RUN_DIR / "environment.json")
    source_hashes = capture_source_hashes()
    write_json(source_hashes, RUN_DIR / "source_hashes.json")

    if not PAIRED_SUBJECTS or len(PAIRED_SUBJECTS) != 24:
        raise ValueError(f"Expected 24 configured subjects, got {len(PAIRED_SUBJECTS)}")
    if len(set(PAIRED_SUBJECTS)) != 24:
        raise ValueError("Duplicate subject IDs in configuration")

    input_audit = {
        "project_root": str(PROJECT_ROOT),
        "expected_subjects": PAIRED_SUBJECTS,
        "data_files": [],
        "split_definitions": [],
        "checkpoint_map": [],
        "baseline_prediction_map": [],
        "validated_subject_inputs": [],
        "missing": [],
        "weighted_v1_directory_exists": WEIGHTED_V1_ROOT.is_dir(),
        "weighted_v1_script_exists": (PROJECT_ROOT / "scripts" / "decision_level_fusion" / "22_train_validation_weighted_fusion.py").is_file(),
    }
    baseline_summary_path = BASELINE_ROOT / "metrics" / "late_fusion_summary.csv"
    if not baseline_summary_path.is_file():
        raise FileNotFoundError(f"Saved 50:50 summary missing: {baseline_summary_path}")
    input_audit["baseline_summary_path"] = str(baseline_summary_path)
    input_audit["baseline_summary_sha256"] = sha256(baseline_summary_path)
    split_dir = MODEL_RESULT_ROOT / "splits"
    checkpoint_dir = MODEL_RESULT_ROOT / "checkpoints"
    prediction_dir = BASELINE_ROOT / "predictions"
    for index, test_subject in enumerate(PAIRED_SUBJECTS):
        fold = index + 1
        validation_subject = PAIRED_SUBJECTS[(index + 1) % 24]
        split_path = split_dir / f"fold_{fold:02d}.json"
        checkpoint_eeg = checkpoint_dir / f"eeg_fold_{fold:02d}_best.pt"
        checkpoint_fnirs = checkpoint_dir / f"fnirs_fold_{fold:02d}_best.pt"
        test_prediction = prediction_dir / f"{test_subject}_late_fusion_predictions.csv"
        data_path = Path(DATA_DIR) / f"{test_subject}_nback_preprocessed.mat"
        expected_paths = [split_path, checkpoint_eeg, checkpoint_fnirs, test_prediction, data_path]
        missing = [str(path) for path in expected_paths if not path.is_file()]
        if missing:
            input_audit["missing"].extend(missing)
            raise FileNotFoundError("Required inputs missing:\n" + "\n".join(missing))
        split = json.loads(split_path.read_text(encoding="utf-8"))
        wanted_train = expected_train(test_subject, validation_subject)
        if split.get("test_subject") != test_subject or split.get("val_subject") != validation_subject or split.get("train_subjects") != wanted_train:
            raise ValueError(f"LOSO split definition mismatch: {split_path}")
        eeg_checkpoint = torch.load(checkpoint_eeg, map_location="cpu", weights_only=False)
        fnirs_checkpoint = torch.load(checkpoint_fnirs, map_location="cpu", weights_only=False)
        if eeg_checkpoint.get("fold", fold) != fold or fnirs_checkpoint.get("fold", fold) != fold:
            raise ValueError(f"Checkpoint-to-fold mismatch at fold {fold}")
        input_audit["split_definitions"].append({
            "fold": fold, "path": str(split_path), "train_subjects": split["train_subjects"],
            "validation_subject": validation_subject, "test_subject": test_subject,
        })
        input_audit["checkpoint_map"].append({
            "fold": fold,
            "test_subject": test_subject,
            "eeg": {"path": str(checkpoint_eeg), "bytes": checkpoint_eeg.stat().st_size, "sha256": sha256(checkpoint_eeg), "normalization_stats_present": "train_mean" in eeg_checkpoint and "train_std" in eeg_checkpoint},
            "fnirs": {"path": str(checkpoint_fnirs), "bytes": checkpoint_fnirs.stat().st_size, "sha256": sha256(checkpoint_fnirs), "normalization_stats_present": "train_mean" in fnirs_checkpoint and "train_std" in fnirs_checkpoint},
        })
        baseline_frame, baseline_info = read_saved_test(test_subject)
        input_audit["baseline_prediction_map"].append({"fold": fold, "test_subject": test_subject, **baseline_info, "rows": len(baseline_frame)})
        input_audit["data_files"].append({"subject": test_subject, "path": str(data_path), "bytes": data_path.stat().st_size})

    write_json(input_audit, RUN_DIR / "input_audit.json")
    log(f"Preflight passed for {len(PAIRED_SUBJECTS)} subjects; using device={device}")

    baseline_rows: list[dict] = []
    candidate_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    selection_rows: list[dict] = []
    baseline_predictions: list[dict] = []
    candidate_predictions: list[dict] = []
    validation_labels_by_subject = {}
    cached_test_frames = {}
    feature_seconds = 0.0
    status = "RUNNING"
    failure = None

    try:
        for index, test_subject in enumerate(PAIRED_SUBJECTS):
            check_deadline(f"fold {index + 1}")
            fold = index + 1
            validation_subject = PAIRED_SUBJECTS[(index + 1) % 24]
            split = input_audit["split_definitions"][index]
            val_started = time.monotonic()
            eeg_ckpt = checkpoint_dir / f"eeg_fold_{fold:02d}_best.pt"
            fnirs_ckpt = checkpoint_dir / f"fnirs_fold_{fold:02d}_best.pt"
            validation_data = load_subject(validation_subject)
            eeg_val_logits, val_labels, eeg_audit = load_validation_inference("eeg", eeg_ckpt, fold, validation_subject, validation_data, device)
            fnirs_val_logits, fnirs_labels, fnirs_audit = load_validation_inference("fnirs", fnirs_ckpt, fold, validation_subject, validation_data, device)
            feature_seconds += time.monotonic() - val_started
            if not np.array_equal(val_labels, fnirs_labels):
                raise ValueError(f"EEG/fNIRS label alignment error for validation subject {validation_subject}")
            if not np.array_equal(eeg_audit["trial_ids"], fnirs_audit["trial_ids"]):
                raise ValueError(f"EEG/fNIRS trial alignment error for validation subject {validation_subject}")
            validation_labels_by_subject[validation_subject] = {
                "labels": val_labels.tolist(), "trial_ids": eeg_audit["trial_ids"],
                "subject_ids": eeg_audit["subject_ids"],
            }
            input_audit["validated_subject_inputs"].append({
                "fold": fold, "validation_subject": validation_subject,
                "eeg_shape": eeg_audit["eeg_shape"], "fnirs_shape": fnirs_audit["fnirs_shape"],
                "n_samples": 27, "class_counts": {"LOW": 9, "HIGH": 18},
                "trial_ids_aligned": True, "labels_aligned": True,
                "eeg_checkpoint": str(eeg_ckpt), "fnirs_checkpoint": str(fnirs_ckpt),
            })
            eeg_val_probability = predict_from_logits(eeg_val_logits)
            fnirs_val_probability = predict_from_logits(fnirs_val_logits)
            equal_val = (eeg_val_probability + fnirs_val_probability) / 2
            eeg_temp, eeg_cal_val = fit_temperature(eeg_val_logits, val_labels)
            fnirs_temp, fnirs_cal_val = fit_temperature(fnirs_val_logits, val_labels)
            calibrated_eeg_val = predict_from_logits(eeg_val_logits, eeg_temp)
            calibrated_fnirs_val = predict_from_logits(fnirs_val_logits, fnirs_temp)
            calibrated_equal_val = (calibrated_eeg_val + calibrated_fnirs_val) / 2
            threshold, val_selected_metrics, val_default_metrics, val_best_threshold, val_best_metrics, threshold_accepted = conservative_threshold(val_labels, equal_val)
            calibrated_threshold, val_cal_selected, val_cal_default, val_cal_best_threshold, val_cal_best, calibrated_threshold_accepted = conservative_threshold(val_labels, calibrated_equal_val)

            # Freeze every parameter before reading held-out labels for evaluation.
            frame, _ = read_saved_test(test_subject)
            cached_test_frames[test_subject] = frame
            p_eeg = frame["eeg_probability_1"].to_numpy(dtype=np.float64)
            p_fnirs = frame["fnirs_probability_1"].to_numpy(dtype=np.float64)
            p_equal = average_probability_fusion(
                torch.from_numpy(frame[["eeg_probability_0", "eeg_probability_1"]].to_numpy(dtype=np.float32)),
                torch.from_numpy(frame[["fnirs_probability_0", "fnirs_probability_1"]].to_numpy(dtype=np.float32)),
            ).numpy()[:, 1].astype(np.float64)
            test_logits_eeg = np.stack([np.log1p(-np.clip(p_eeg, 1e-7, 1 - 1e-7)), np.log(np.clip(p_eeg, 1e-7, 1 - 1e-7))], axis=1)
            test_logits_fnirs = np.stack([np.log1p(-np.clip(p_fnirs, 1e-7, 1 - 1e-7)), np.log(np.clip(p_fnirs, 1e-7, 1 - 1e-7))], axis=1)
            calibrated_eeg = predict_from_logits(test_logits_eeg, eeg_temp)
            calibrated_fnirs = predict_from_logits(test_logits_fnirs, fnirs_temp)
            calibrated_equal = (calibrated_eeg + calibrated_fnirs) / 2
            y_test = frame["true_label"].to_numpy(dtype=np.int64)
            methods = [
                ("baseline_50_50", p_equal, 0.5),
                ("calibrated_equal_50_50", calibrated_equal, 0.5),
                ("conservative_threshold", p_equal, threshold),
                ("calibrated_conservative_threshold", calibrated_equal, calibrated_threshold),
                ("majority_high", np.ones_like(p_equal), 0.5),
            ]
            fold_baseline = None
            for modality, probability, prediction_col in (
                ("EEG", p_eeg, "eeg_prediction"),
                ("fNIRS", p_fnirs, "fnirs_prediction"),
                ("Fusion", p_equal, "fused_prediction"),
            ):
                values = metric_values(y_test, probability)
                values["method"] = modality
                values["fold"] = fold
                values["subject_id"] = test_subject
                values["prediction_source"] = str(BASELINE_ROOT / "predictions" / f"{test_subject}_late_fusion_predictions.csv")
                baseline_rows.append(values)
                if modality == "Fusion":
                    fold_baseline = values
                diagnostic_rows.append({
                    "protocol": "24_subject_LOSO", "fold": fold, "subject_id": test_subject,
                    "method": modality, "source": "cached_heldout_prediction",
                    **{key: values[key] for key in ("brier_score", "log_loss", "ece", "n_samples", "n_low", "n_high")},
                    "mean_p_high_given_low": float(probability[y_test == 0].mean()),
                    "mean_p_high_given_high": float(probability[y_test == 1].mean()),
                })
            for method, probability, method_threshold in methods:
                values = metric_values(y_test, probability, method_threshold)
                values.update({"method": method, "fold": fold, "subject_id": test_subject, "threshold": method_threshold})
                candidate_rows.append(values)
                if method == "baseline_50_50":
                    fold_baseline = values
                for trial_index, trial_id in enumerate(frame["trial_id"].astype(int)):
                    candidate_predictions.append({
                        "fold": fold, "subject_id": test_subject, "trial_id": int(trial_id),
                        "true_label": int(y_test[trial_index]), "method": method,
                        "probability_high": float(probability[trial_index]),
                        "predicted_label": int(probability[trial_index] > method_threshold),
                        "threshold": method_threshold,
                        "eeg_probability_high": float(p_eeg[trial_index]),
                        "fnirs_probability_high": float(p_fnirs[trial_index]),
                    })
            for trial_index, trial_id in enumerate(frame["trial_id"].astype(int)):
                baseline_predictions.append({
                    "fold": fold, "subject_id": test_subject, "trial_id": int(trial_id),
                    "true_label": int(y_test[trial_index]),
                    "eeg_probability_high": float(p_eeg[trial_index]),
                    "fnirs_probability_high": float(p_fnirs[trial_index]),
                    "fused_probability_high": float(p_equal[trial_index]),
                    "eeg_prediction": int(frame["eeg_prediction"].iloc[trial_index]),
                    "fnirs_prediction": int(frame["fnirs_prediction"].iloc[trial_index]),
                    "fused_prediction": int(frame["fused_prediction"].iloc[trial_index]),
                })
            selection_rows.extend([
                {"fold": fold, "test_subject": test_subject, "validation_subject": validation_subject,
                 "parameter": "eeg_temperature", "value": eeg_temp, "selection_metric": "validation_log_loss",
                 "validation_metric": eeg_cal_val["log_loss"], "fit_samples": len(val_labels), "source_checkpoint": str(eeg_ckpt)},
                {"fold": fold, "test_subject": test_subject, "validation_subject": validation_subject,
                 "parameter": "fnirs_temperature", "value": fnirs_temp, "selection_metric": "validation_log_loss",
                 "validation_metric": fnirs_cal_val["log_loss"], "fit_samples": len(val_labels), "source_checkpoint": str(fnirs_ckpt)},
                {"fold": fold, "test_subject": test_subject, "validation_subject": validation_subject,
                 "parameter": "equal_fusion_threshold", "value": threshold, "selection_metric": "validation_macro_f1_guarded",
                 "validation_metric": val_selected_metrics["macro_f1"], "validation_balanced_accuracy": val_selected_metrics["balanced_accuracy"],
                 "default_validation_macro_f1": val_default_metrics["macro_f1"], "best_grid_threshold": val_best_threshold,
                 "best_grid_macro_f1": val_best_metrics["macro_f1"], "parameter_change_accepted": threshold_accepted,
                 "minimum_gain": MIN_VALIDATION_F1_GAIN, "balanced_accuracy_guard": True, "fit_samples": len(val_labels)},
                {"fold": fold, "test_subject": test_subject, "validation_subject": validation_subject,
                 "parameter": "calibrated_fusion_threshold", "value": calibrated_threshold, "selection_metric": "validation_macro_f1_guarded",
                 "validation_metric": val_cal_selected["macro_f1"], "validation_balanced_accuracy": val_cal_selected["balanced_accuracy"],
                 "default_validation_macro_f1": val_cal_default["macro_f1"], "best_grid_threshold": val_cal_best_threshold,
                 "best_grid_macro_f1": val_cal_best["macro_f1"], "parameter_change_accepted": calibrated_threshold_accepted,
                 "minimum_gain": MIN_VALIDATION_F1_GAIN, "balanced_accuracy_guard": True, "fit_samples": len(val_labels)},
            ])
            eeg_error = frame["eeg_prediction"].to_numpy(dtype=int) != y_test
            fnirs_error = frame["fnirs_prediction"].to_numpy(dtype=int) != y_test
            diagnostic_rows.append({
                "protocol": "24_subject_LOSO", "fold": fold, "subject_id": test_subject,
                "method": "modality_error_correlation", "source": "cached_heldout_prediction",
                "error_correlation": float(np.corrcoef(eeg_error, fnirs_error)[0, 1]) if np.std(eeg_error) and np.std(fnirs_error) else float("nan"),
                "eeg_error_count": int(eeg_error.sum()), "fnirs_error_count": int(fnirs_error.sum()),
                "joint_error_count": int((eeg_error & fnirs_error).sum()),
            })
            input_audit["validated_subject_inputs"][-1]["heldout_test_subject"] = test_subject
            save_progress(baseline_rows, candidate_rows, diagnostic_rows, selection_rows, baseline_predictions, candidate_predictions)
            write_json(input_audit, RUN_DIR / "input_audit.json")
            log(f"Fold {fold:02d}/24 completed: validation={validation_subject}; test={test_subject}; threshold={threshold:.2f}; calibrated_threshold={calibrated_threshold:.2f}")

        if set(validation_labels_by_subject) != set(PAIRED_SUBJECTS):
            raise ValueError("Validation loader did not cover each configured subject exactly once")
        for subject in PAIRED_SUBJECTS:
            frame = cached_test_frames[subject]
            validated = validation_labels_by_subject[subject]
            if frame["trial_id"].astype(int).tolist() != validated["trial_ids"]:
                raise ValueError(f"Saved test trial IDs differ from loaded data for {subject}")
            if frame["true_label"].astype(int).tolist() != validated["labels"]:
                raise ValueError(f"Saved test labels differ from loaded data for {subject}")
        input_audit["coverage_verified"] = True
        input_audit["coverage_samples"] = len(baseline_predictions)
        input_audit["all_subjects_seen_as_validation_and_test"] = True
        recomputed_fusion = next(row for row in aggregate(baseline_rows, "method") if row["method"] == "Fusion")
        saved_summary = pd.read_csv(baseline_summary_path).iloc[0]
        baseline_comparison = []
        for metric, saved_key in (
            ("accuracy", "accuracy_mean"),
            ("macro_f1", "macro_f1_mean"),
            ("roc_auc", "roc_auc_mean"),
        ):
            actual = float(recomputed_fusion[f"{metric}_mean"])
            saved = float(saved_summary[saved_key])
            baseline_comparison.append({
                "metric": metric,
                "saved_summary": saved,
                "recomputed_from_predictions": actual,
                "absolute_difference": abs(saved - actual),
                "matches_within_1e-10": abs(saved - actual) < 1e-10,
            })
        input_audit["baseline_summary_verification"] = baseline_comparison
        write_json(input_audit, RUN_DIR / "input_audit.json")
        status = "COMPLETE"
    except DeadlineReached as error:
        status = "INCOMPLETE"
        failure = str(error)
        log(failure)
    except Exception as error:
        status = "FAILED"
        failure = f"{type(error).__name__}: {error}"
        log(failure)
        raise
    finally:
        runtime = time.monotonic() - EXPERIMENT_STARTED
        manifest = {
            "experiment_id": RUN_DIR.name,
            "status": status,
            "started_at_utc": EXPERIMENT_STARTED_AT.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": runtime,
            "workflow_started_at": WORKFLOW_STARTED_AT.isoformat(),
            "workflow_deadline": WORKFLOW_DEADLINE.isoformat(),
            "total_workflow_elapsed_seconds": (datetime.now(timezone.utc) - WORKFLOW_STARTED_AT.astimezone(timezone.utc)).total_seconds(),
            "validation_data_and_inference_seconds": feature_seconds,
            "wall_clock_limit_seconds": MAX_RUNTIME_SECONDS,
            "completed_folds": len({row["fold"] for row in candidate_rows if row["method"] == "baseline_50_50"}),
            "evaluation_samples": len(baseline_predictions),
            "device": str(device),
            "failure_or_stop_reason": failure,
            "optional_meta_fusion": "SKIPPED: no independent training-side OOF probabilities saved",
            "optional_architecture": "SKIPPED: primary decision-fusion diagnostics prioritized; no new architecture was trained or claimed",
        }
        write_json(manifest, RUN_DIR / "run_manifest.json")
        save_progress(baseline_rows, candidate_rows, diagnostic_rows, selection_rows, baseline_predictions, candidate_predictions)

    if status == "COMPLETE":
        summaries = make_summary(candidate_rows)
        comparisons = paired_comparisons(
            [row for row in candidate_rows if row["method"] != "baseline_50_50"],
            [row for row in candidate_rows if row["method"] == "baseline_50_50"],
        )
        write_csv(summaries, RUN_DIR / "candidate_summary.csv")
        write_csv(comparisons, RUN_DIR / "fold_comparison.csv")
        findings = [
            {"issue": "Class imbalance", "evidence": "Each subject has 9 LOW and 18 HIGH trials; majority-HIGH reference is 66.67% accuracy.", "severity": "High", "status": "Confirmed", "fix": "Added class-wise metrics and majority baseline; no training change made."},
            {"issue": "Validation threshold instability", "evidence": "Prior broad threshold tuning lowered LOSO Macro-F1; this run used a three-value grid plus +0.05 F1 and balanced-accuracy guards.", "severity": "Moderate", "status": "Confirmed", "fix": "Conservative validation-only threshold rule; defaults retained unless guard passes."},
            {"issue": "Calibration sample size", "evidence": "One validation subject contributes 27 trials per fold; temperature is selected from five fixed values.", "severity": "Moderate", "status": "Limitation", "fix": "No isotonic calibration; calibrated results remain exploratory."},
            {"issue": "Weighted v1 provenance", "evidence": f"Expected directory exists={WEIGHTED_V1_ROOT.is_dir()}; expected training script exists={(PROJECT_ROOT / 'scripts' / 'decision_level_fusion' / '22_train_validation_weighted_fusion.py').is_file()}.", "severity": "Low", "status": "Confirmed missing", "fix": "Marked unavailable; prior v2 weighting artifacts separately recorded."},
            {"issue": "Checkpoint normalization and fold map", "evidence": "Each fold-specific checkpoint is loaded by fold ID and its stored training mean/std are applied to validation data.", "severity": "High", "status": "Verified", "fix": "No source checkpoint or model changes."},
            {"issue": "Trial/label alignment", "evidence": "All baseline prediction rows were checked against MAT trial IDs and labels after each subject was loaded as a validation subject.", "severity": "High", "status": "Verified", "fix": "Mismatches would stop the run; all 648 rows aligned."},
            {"issue": "Cross-modal error correlation", "evidence": "Per-fold correlation and joint error counts are saved in prediction_diagnostics.csv.", "severity": "Diagnostic", "status": "Measured", "fix": "No change applied."},
        ]
        write_csv(findings, RUN_DIR / "diagnostic_findings.csv")
        prior_summary = PRIOR_V2_ROOT / "loso_summary.csv"
        baseline_summary = aggregate(baseline_rows, "method")
        workflow_runtime = (datetime.now(timezone.utc) - WORKFLOW_STARTED_AT.astimezone(timezone.utc)).total_seconds()
        report = make_report(status, input_audit, environment, baseline_summary, summaries, comparisons, findings, workflow_runtime, feature_seconds)
        if prior_summary.is_file():
            prior = pd.read_csv(prior_summary)
            prior_lines = ["", "## Previous Validation-Only Run", "", f"The earlier run at `{PRIOR_V2_ROOT}` reported threshold-only Macro-F1 0.5903 and weighted-fusion Macro-F1 0.5852 versus 0.6226 for 50:50. This run tests calibration and a narrower, guarded threshold decision rather than repeating the earlier broad threshold search.", ""]
            report += "\n" + "\n".join(prior_lines)
        report += "\n## Exact Output Paths\n\n" + "\n".join(f"- `{RUN_DIR / name}`" for name in (
            "config.json", "input_audit.json", "environment.json", "source_hashes.json", "baseline_metrics.csv",
            "candidate_fold_metrics.csv", "candidate_summary.csv", "fold_comparison.csv", "selected_parameters.csv",
            "prediction_diagnostics.csv", "diagnostic_findings.csv", "run_log.txt", "run_manifest.json",
        )) + "\n"
        (RUN_DIR / "final_report.md").write_text(report, encoding="utf-8")
    else:
        (RUN_DIR / "final_report.md").write_text(
            f"# Incomplete/failed decision-fusion run\n\nStatus: {status}\nCompleted folds: {manifest['completed_folds']}/24\nReason: {failure}\n\nPartial fold artifacts are not a full LOSO score.\n",
            encoding="utf-8",
        )

    log(f"Run finished with status={status}; runtime={runtime:.1f}s; validation inference={feature_seconds:.1f}s; folds={manifest['completed_folds']}/24")
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())