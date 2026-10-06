from __future__ import annotations

import json
import os
import platform
import random
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from src.fnirs.features import extract_fnirs_features
from src.fusion.alignment import align_modalities
from src.fusion.data import apply_binary_workload_labels, load_fnirs_provenance, load_modality_data
from src.fusion.features import extract_eeg_features


LABELS = [0, 1]
LABEL_NAMES = ["Low workload / 0-back", "High workload / 2+3-back"]


def _leave_one_subject_out_folds(subject_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    subject_ids = np.asarray(subject_ids)
    if subject_ids.ndim != 1 or len(np.unique(subject_ids)) < 2:
        raise ValueError("LOSO requires at least two subject IDs in a one-dimensional array.")
    splitter = LeaveOneGroupOut()
    folds = list(splitter.split(np.zeros(len(subject_ids)), groups=subject_ids))
    for train_idx, test_idx in folds:
        if set(subject_ids[train_idx]) & set(subject_ids[test_idx]) or len(np.unique(subject_ids[test_idx])) != 1:
            raise ValueError("LOSO split contains subject leakage or does not hold out exactly one subject.")
    return folds


def _pipeline(model_name: str, params: dict[str, Any], seed: int) -> Pipeline:
    if model_name == "linear_svm":
        classifier = LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000, **params)
    elif model_name == "rbf_svm":
        classifier = SVC(kernel="rbf", class_weight="balanced", **params)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def _select_svm_params(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    candidates: list[dict[str, Any]],
    seed: int,
    inner_folds: int,
) -> tuple[dict[str, Any], float]:
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 3:
        return candidates[0], float("nan")
    splitter = GroupKFold(n_splits=min(inner_folds, len(unique_subjects)))
    usable_splits = [
        (train, valid)
        for train, valid in splitter.split(X, y, groups=subjects)
        if len(np.unique(y[train])) == 2 and len(np.unique(y[valid])) == 2
    ]
    if not usable_splits:
        return candidates[0], float("nan")

    scores: list[tuple[float, dict[str, Any]]] = []
    for params in candidates:
        fold_scores = []
        for train_idx, valid_idx in usable_splits:
            candidate = _pipeline(model_name, params, seed)
            candidate.fit(X[train_idx], y[train_idx])
            fold_scores.append(balanced_accuracy_score(y[valid_idx], candidate.predict(X[valid_idx])))
        scores.append((float(np.mean(fold_scores)), params))
    best_score, best_params = max(scores, key=lambda item: item[0])
    return best_params, best_score


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, scores))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "roc_auc": auc,
    }


def _positive_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    class_one_index = int(np.where(classifier.classes_ == 1)[0][0])
    return np.asarray(model.predict_proba(X)[:, class_one_index], dtype=float)


def _evaluate_view(
    view: str,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["random_state"])
    candidate_configs = config["models"]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    folds = _leave_one_subject_out_folds(subjects)
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_subjects = set(subjects[train_idx].tolist())
        test_subjects = set(subjects[test_idx].tolist())
        if train_subjects & test_subjects or len(test_subjects) != 1:
            raise ValueError("LOSO split contains subject leakage or does not hold out exactly one subject.")
        if len(np.unique(y[train_idx])) != 2:
            raise ValueError(f"Cannot train fold {fold_idx}: training subjects contain only one workload class.")

        for model_name in ("linear_svm", "rbf_svm", "random_forest"):
            settings = candidate_configs[model_name]
            if model_name == "random_forest":
                chosen_params = dict(settings["parameters"])
                inner_score = float("nan")
            else:
                candidates = list(ParameterGrid(settings["param_grid"]))
                chosen_params, inner_score = _select_svm_params(
                    model_name,
                    X[train_idx],
                    y[train_idx],
                    subjects[train_idx],
                    candidates,
                    seed,
                    int(config.get("inner_group_folds", 3)),
                )
            model = _pipeline(model_name, chosen_params, seed)
            model.fit(X[train_idx], y[train_idx])
            y_test = y[test_idx]
            predictions = model.predict(X[test_idx]).astype(int)
            scores = _positive_scores(model, X[test_idx])
            test_subject = str(subjects[test_idx][0])
            fold_rows.append({
                "view": view,
                "model": model_name,
                "fold": fold_idx,
                "test_subject": test_subject,
                "n_test_epochs": int(len(test_idx)),
                **_binary_metrics(y_test, predictions, scores),
                "inner_group_cv_balanced_accuracy": inner_score,
                "selected_parameters": json.dumps(chosen_params, sort_keys=True),
            })
            for idx, true_label, prediction, score in zip(test_idx, y_test, predictions, scores):
                prediction_rows.append({
                    "view": view,
                    "model": model_name,
                    "fold": fold_idx,
                    "subject_id": str(subjects[idx]),
                    "true_label": int(true_label),
                    "predicted_label": int(prediction),
                    "score_high_workload": float(score),
                })
    return fold_rows, prediction_rows


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for values, suffix, fmt in ((cm, "", "d"), (normalized, "_normalized", ".2f")):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        image = ax.imshow(values, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(LABELS, LABEL_NAMES, rotation=15, ha="right")
        ax.set_yticks(LABELS, LABEL_NAMES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, format(values[row, col], fmt), ha="center", va="center", color="black")
        fig.tight_layout()
        root, ext = os.path.splitext(path)
        if suffix:
            normalized_root = root.replace("_confusion_matrix", "_normalized_confusion_matrix")
            fig.savefig(f"{normalized_root}{ext}", dpi=180)
        else:
            fig.savefig(path, dpi=180)
        plt.close(fig)


def _write_outputs(output_root: str, fold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> None:
    os.makedirs(output_root, exist_ok=True)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    folds.to_csv(os.path.join(output_root, "loso_subject_metrics.csv"), index=False)
    predictions.to_csv(os.path.join(output_root, "out_of_fold_predictions.csv"), index=False)

    pooled_rows = []
    for (view, model_name), group in predictions.groupby(["view", "model"], sort=True):
        y_true = group["true_label"].to_numpy(dtype=int)
        y_pred = group["predicted_label"].to_numpy(dtype=int)
        scores = group["score_high_workload"].to_numpy(dtype=float)
        fold_subset = folds[(folds["view"] == view) & (folds["model"] == model_name)]
        pooled_rows.append({
            "view": view,
            "model": model_name,
            "n_subjects": int(fold_subset["test_subject"].nunique()),
            "n_test_epochs": int(len(group)),
            **_binary_metrics(y_true, y_pred, scores),
            "accuracy_subject_mean": float(fold_subset["accuracy"].mean()),
            "accuracy_subject_std": float(fold_subset["accuracy"].std(ddof=0)),
            "balanced_accuracy_subject_mean": float(fold_subset["balanced_accuracy"].mean()),
            "balanced_accuracy_subject_std": float(fold_subset["balanced_accuracy"].std(ddof=0)),
            "macro_f1_subject_mean": float(fold_subset["macro_f1"].mean()),
            "macro_f1_subject_std": float(fold_subset["macro_f1"].std(ddof=0)),
            "roc_auc_subject_mean": float(fold_subset["roc_auc"].mean()),
            "roc_auc_subject_std": float(fold_subset["roc_auc"].std(ddof=0)),
        })
        _plot_confusion(y_true, y_pred, os.path.join(output_root, f"{view}_{model_name}_confusion_matrix.png"), f"{view}: {model_name}")
    pd.DataFrame(pooled_rows).to_csv(os.path.join(output_root, "modality_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    comparison = pd.DataFrame(pooled_rows)
    labels = comparison["view"] + "\n" + comparison["model"]
    ax.bar(labels, comparison["macro_f1_subject_mean"], yerr=comparison["macro_f1_subject_std"], capsize=3, color="#356B5A")
    ax.set_ylabel("Subject-level Macro-F1 (mean +/- SD)")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(os.path.join(output_root, "modality_comparison.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = 0
    for (view, model_name), group in predictions.groupby(["view", "model"], sort=True):
        labels = group["true_label"].to_numpy(dtype=int)
        if len(np.unique(labels)) != 2:
            continue
        fpr, tpr, _ = roc_curve(labels, group["score_high_workload"].to_numpy(dtype=float))
        auc = roc_auc_score(labels, group["score_high_workload"].to_numpy(dtype=float))
        ax.plot(fpr, tpr, label=f"{view} / {model_name} (AUC {auc:.3f})")
        plotted += 1
    if plotted:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Pooled out-of-fold ROC curves")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_root, "roc_curves.png"), dpi=180)
    plt.close(fig)


def _save_alignment(output_dir: str, aligned: pd.DataFrame, report: dict[str, Any], eeg_meta: pd.DataFrame, fnirs_meta: pd.DataFrame) -> None:
    os.makedirs(output_dir, exist_ok=True)
    aligned.to_csv(os.path.join(output_dir, "aligned_metadata.csv"), index=False)
    details = []
    for modality, metadata, other in (("eeg", eeg_meta, fnirs_meta), ("fnirs", fnirs_meta, eeg_meta)):
        other_keys = set(zip(other["subject_id"], other["trial_id"]))
        for row in metadata.itertuples(index=False):
            key = (row.subject_id, row.trial_id)
            if key not in other_keys:
                details.append({"modality": modality, "subject_id": key[0], "trial_id": key[1], "status": f"{modality}_only"})
    counts = pd.DataFrame([{"record_type": "count", "modality": key, "count": report[key]} for key in (
        "eeg_trials", "fnirs_trials", "matched_trials", "unmatched_eeg_trials", "unmatched_fnirs_trials"
    )])
    details_frame = pd.DataFrame(details, columns=["modality", "subject_id", "trial_id", "status"])
    counts.to_csv(os.path.join(output_dir, "alignment_report.csv"), index=False)
    details_frame.to_csv(os.path.join(output_dir, "unmatched_trials.csv"), index=False)


def run_experiment(config_path: str = "configs/workload_fusion.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    _, fnirs_provenance = load_fnirs_provenance(config)
    seed = int(config["random_state"])
    random.seed(seed)
    np.random.seed(seed)

    eeg_X, eeg_raw, eeg_report = load_modality_data(config["data"]["eeg_dir"], "EEG")
    fnirs_X, fnirs_raw, fnirs_report = load_modality_data(config["data"]["fnirs_dir"], "fNIRS")
    eeg_meta = apply_binary_workload_labels(eeg_raw, config["labels"], "EEG", eeg_report["label_map"])
    fnirs_meta = apply_binary_workload_labels(fnirs_raw, config["labels"], "fNIRS", fnirs_report["label_map"])

    print("DATA AUDIT")
    for report in (eeg_report, fnirs_report):
        print(f"{report['modality']} subjects={report['n_subjects']} epochs={report['n_epochs']} channels={report['n_channels']} timepoints={report['n_timepoints']}")
        for source_code, semantic_name in sorted(config["labels"]["source_label_names"].items(), key=lambda item: int(item[0])):
            count = int((report["class_distribution"].get(str(int(source_code)), 0)))
            print(f"  original {semantic_name}: {count}")
        binary_counts = eeg_meta if report["modality"] == "EEG" else fnirs_meta
        print(f"  binary low workload={int((binary_counts['binary_label'] == 0).sum())}, high workload={int((binary_counts['binary_label'] == 1).sum())}")
    print(f"fNIRS representation={fnirs_provenance['representation']}")
    print(f"Subjects listed in both modalities={len(set(eeg_meta.subject_id) & set(fnirs_meta.subject_id))}")

    eeg_alignment = eeg_meta[["subject_id", "trial_id", "binary_label"]].rename(columns={"binary_label": "label"})
    fnirs_alignment = fnirs_meta[["subject_id", "trial_id", "binary_label"]].rename(columns={"binary_label": "label"})
    aligned, alignment_report = align_modalities(eeg_alignment, fnirs_alignment)
    if aligned.empty:
        raise ValueError("No EEG-fNIRS subject-trial pairs matched; training stopped.")
    if aligned["subject_id"].nunique() < 2:
        raise ValueError("At least two paired subjects are required for subject-independent evaluation.")
    if set(aligned["eeg_label"].astype(int).unique()) != {0, 1}:
        raise ValueError("Aligned trials do not contain both workload classes; training stopped.")
    print(f"Paired subjects with aligned trials={aligned['subject_id'].nunique()}")
    aligned = aligned.rename(columns={"eeg_label": "binary_label"})
    eeg_lookup = {(row.subject_id, row.trial_id): index for index, row in enumerate(eeg_meta.itertuples(index=False))}
    fnirs_lookup = {(row.subject_id, row.trial_id): index for index, row in enumerate(fnirs_meta.itertuples(index=False))}
    keys = list(zip(aligned["subject_id"], aligned["trial_id"]))
    eeg_paired_idx = np.asarray([eeg_lookup[key] for key in keys], dtype=int)
    fnirs_paired_idx = np.asarray([fnirs_lookup[key] for key in keys], dtype=int)
    paired_subjects = aligned["subject_id"].astype(str).to_numpy()
    paired_y = aligned["binary_label"].astype(int).to_numpy()
    output_dirs = config["outputs"]
    _save_alignment(output_dirs["fusion"], aligned, alignment_report, eeg_meta, fnirs_meta)

    eeg_channels = eeg_report["channel_names"] or [f"ch_{idx}" for idx in range(eeg_X.shape[1])]
    if len(eeg_channels) != eeg_X.shape[1]:
        raise ValueError("EEG channel name count does not match the EEG channel dimension.")
    eeg_features = extract_eeg_features(
        eeg_X,
        channel_names=eeg_channels,
        sampling_frequency=float(config["eeg"]["sampling_rate_hz"]),
        bands={name: tuple(values) for name, values in config["eeg"]["frequency_bands_hz"].items()},
    )
    fnirs_features, _ = extract_fnirs_features(fnirs_X)

    views = [
        ("eeg_all", eeg_features, eeg_meta["binary_label"].to_numpy(dtype=int), eeg_meta["subject_id"].to_numpy(dtype=str), None),
        ("eeg_paired", eeg_features[eeg_paired_idx], paired_y, paired_subjects, "eeg"),
        ("fnirs_paired", fnirs_features[fnirs_paired_idx], paired_y, paired_subjects, "fnirs"),
        ("early_fusion", np.concatenate([eeg_features[eeg_paired_idx], fnirs_features[fnirs_paired_idx]], axis=1), paired_y, paired_subjects, "fusion"),
    ]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for view, features, labels, subjects, _ in views:
        print(f"Evaluating {view}: {len(labels)} epochs, {len(np.unique(subjects))} LOSO folds")
        fold_results, predictions = _evaluate_view(view, features, labels, subjects, config)
        fold_rows.extend(fold_results)
        prediction_rows.extend(predictions)

    comparison_dir = output_dirs["comparison"]
    _write_outputs(comparison_dir, fold_rows, prediction_rows)
    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    for directory, views_to_keep in (
        (output_dirs["eeg"], {"eeg_all", "eeg_paired"}),
        (output_dirs["fnirs"], {"fnirs_paired"}),
        (output_dirs["fusion"], {"early_fusion"}),
    ):
        os.makedirs(directory, exist_ok=True)
        fold_frame[fold_frame["view"].isin(views_to_keep)].to_csv(os.path.join(directory, "loso_subject_metrics.csv"), index=False)
        prediction_frame[prediction_frame["view"].isin(views_to_keep)].to_csv(os.path.join(directory, "out_of_fold_predictions.csv"), index=False)
        pd.read_csv(os.path.join(comparison_dir, "modality_comparison.csv")).query("view in @views_to_keep").to_csv(os.path.join(directory, "metrics.csv"), index=False)
        for _, group in prediction_frame[prediction_frame["view"].isin(views_to_keep)].groupby(["view", "model"]):
            name = f"{group['view'].iloc[0]}_{group['model'].iloc[0]}"
            _plot_confusion(group["true_label"].to_numpy(), group["predicted_label"].to_numpy(), os.path.join(directory, f"{name}_confusion_matrix.png"), name)
        reference_view = {output_dirs["eeg"]: "eeg_paired", output_dirs["fnirs"]: "fnirs_paired", output_dirs["fusion"]: "early_fusion"}[directory]
        reference = prediction_frame[
            (prediction_frame["view"] == reference_view) & (prediction_frame["model"] == "random_forest")
        ]
        _plot_confusion(
            reference["true_label"].to_numpy(),
            reference["predicted_label"].to_numpy(),
            os.path.join(directory, "confusion_matrix.png"),
            f"{reference_view}: random_forest reference",
        )

    comparison = pd.read_csv(os.path.join(comparison_dir, "modality_comparison.csv"))
    comparison[comparison["view"].isin({"eeg_all", "eeg_paired", "fnirs_paired", "early_fusion"})].to_csv(os.path.join(comparison_dir, "modality_comparison.csv"), index=False)
    fold_frame.to_csv(os.path.join(comparison_dir, "per_subject_metrics.csv"), index=False)
    with open(os.path.join(comparison_dir, "config_used.yaml"), "w", encoding="utf8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    package_versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not installed"
    with open(os.path.join(comparison_dir, "reproducibility.json"), "w", encoding="utf8") as handle:
        json.dump({"python": platform.python_version(), "packages": package_versions, "random_state": seed, "fnirs_preprocessing": fnirs_provenance}, handle, indent=2)

    eeg_subjects = set(eeg_meta.subject_id)
    fnirs_subjects = set(fnirs_meta.subject_id)
    aligned_subjects = set(aligned["subject_id"].astype(str))
    subject_sets = {
        "eeg_without_aligned_trials": sorted(eeg_subjects - aligned_subjects),
        "fnirs_without_aligned_trials": sorted(fnirs_subjects - aligned_subjects),
        "subjects_in_both_but_without_matched_trials": sorted((eeg_subjects & fnirs_subjects) - aligned_subjects),
        "paired_with_aligned_trials": sorted(aligned_subjects),
    }
    with open(os.path.join(comparison_dir, "subject_sets.json"), "w", encoding="utf8") as handle:
        json.dump(subject_sets, handle, indent=2)
    summary = pd.read_csv(os.path.join(comparison_dir, "modality_comparison.csv"))
    summary_lines = [
        "Binary cognitive workload classification (subject-independent LOSO)",
        f"EEG epochs: {eeg_report['n_epochs']} across {eeg_report['n_subjects']} subjects",
        f"fNIRS epochs: {fnirs_report['n_epochs']} across {fnirs_report['n_subjects']} subjects",
        f"Paired subjects with aligned trials: {len(subject_sets['paired_with_aligned_trials'])}",
        f"Matched subject-trial epochs: {alignment_report['matched_trials']}",
        f"Unmatched EEG/fNIRS trials: {alignment_report['unmatched_eeg_trials']}/{alignment_report['unmatched_fnirs_trials']}",
        f"Binary counts, EEG: low={(eeg_meta['binary_label'] == 0).sum()}, high={(eeg_meta['binary_label'] == 1).sum()}",
        f"Binary counts, fNIRS: low={(fnirs_meta['binary_label'] == 0).sum()}, high={(fnirs_meta['binary_label'] == 1).sum()}",
        "",
        summary[["view", "model", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1", "roc_auc"]].to_string(index=False),
        "",
        "Pooled metrics use out-of-fold trial predictions; subject variability is in per_subject_metrics.csv.",
        "ROC-AUC is NaN when its evaluation set contains only one class.",
    ]
    with open(os.path.join(comparison_dir, "summary.txt"), "w", encoding="utf8") as handle:
        handle.write("\n".join(summary_lines) + "\n")
    print("SUBJECT EXCLUSIONS/PAIRED SETS", json.dumps(subject_sets))
    print(f"Aligned trials={alignment_report['matched_trials']}; unmatched EEG={alignment_report['unmatched_eeg_trials']}; unmatched fNIRS={alignment_report['unmatched_fnirs_trials']}")
    summary = pd.read_csv(os.path.join(comparison_dir, "modality_comparison.csv"))
    print("FINAL POOLED OUT-OF-FOLD METRICS")
    print(summary[["view", "model", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1", "roc_auc"]].to_string(index=False))
    print("Subject-level variability is saved in modality_comparison.csv and per_subject_metrics.csv.")
    print(f"Results saved under {os.path.abspath(os.path.dirname(comparison_dir))}")
    return {"eeg": eeg_report, "fnirs": fnirs_report, "alignment": alignment_report, "folds": fold_frame, "predictions": prediction_frame}