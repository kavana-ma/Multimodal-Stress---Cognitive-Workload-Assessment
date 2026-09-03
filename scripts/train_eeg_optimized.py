from pathlib import Path
import sys
import shutil
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eeg.data.loader import load_eeg_dataset
from src.eeg.features import extract_features, extract_optimized_features


def score(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred), "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0), "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0), "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/eeg_baseline.yaml").read_text(encoding="utf-8"))
    np.random.seed(cfg["runtime"]["seed"])
    data = load_eeg_dataset(root / cfg["dataset"]["path"])
    X, y, groups = data["X"], data["y"], data["subject_id"]
    fs = data["preprocessing_config"]["sampling_rate_hz"]
    baseline, baseline_names = extract_features(X, fs, cfg["features"]["bands"], cfg["features"]["welch_nperseg"])
    improved_bands = {"delta": (1, 4), **cfg["features"]["bands"]}
    improved, improved_names = extract_optimized_features(X, data["channel_names"], fs, improved_bands, cfg["features"]["welch_nperseg"])
    if not np.isfinite(improved).all():
        raise ValueError("Invalid optimized feature values")
    out = root / "results/eeg_baseline/improved"; out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    pd.DataFrame({"feature_name": improved_names}).to_csv(out / "feature_names.csv", index=False)
    labels = np.unique(y); class_names = [data["label_map"].get(f"class_{label}", str(label)) for label in labels]
    pd.DataFrame({"label": labels, "class_name": class_names, "count": [(y == label).sum() for label in labels]}).to_csv(out / "class_distribution.csv", index=False)
    pd.DataFrame({"subject_id": np.unique(groups), "epochs": [int((groups == subject).sum()) for subject in np.unique(groups)]}).to_csv(out / "epochs_per_subject.csv", index=False)
    (out / "data_qc.txt").write_text("EOG channels used: none; input channel count: 28.\nEOG rejection mask: unavailable in repository metadata; no epochs removed.\nTrial IDs repeat across subjects but (subject_id, trial_id) pairs are unique.\n", encoding="utf-8")
    e0 = pd.read_csv(root / "results/eeg_baseline/model_comparison.csv").rename(columns={"Model": "Model"})
    e0["Experiment"] = "E0 baseline"; e0["Model"] = "E0 " + e0["Model"]
    models = {
        "Optimized Linear SVM": (Pipeline([("scale", StandardScaler()), ("variance", VarianceThreshold()), ("select", SelectKBest(f_classif, k=300)), ("model", SVC(kernel="linear", class_weight="balanced"))]), {"model__C": [0.1, 1, 10]}),
        "Optimized RBF SVM": (Pipeline([("scale", StandardScaler()), ("variance", VarianceThreshold()), ("select", SelectKBest(f_classif, k=300)), ("model", SVC(kernel="rbf", class_weight="balanced"))]), {"model__C": [1, 10, 100], "model__gamma": ["scale", 0.001, 0.01]}),
        "Optimized Random Forest": (Pipeline([("variance", VarianceThreshold()), ("select", SelectKBest(f_classif, k=300)), ("model", RandomForestClassifier(class_weight="balanced", random_state=cfg["runtime"]["seed"], n_jobs=-1))]), {"model__n_estimators": [300], "model__max_depth": [None, 20], "model__min_samples_leaf": [1, 2], "model__max_features": ["sqrt"]})}
    predictions, subject_rows, summaries = [], [], []
    rf_importance = np.zeros(len(improved_names), dtype=float)
    for model_name, (estimator, grid) in models.items():
        model_predictions = []
        print(f"Running {model_name} with {len(np.unique(groups))} LOSO folds")
        for fold, subject in enumerate(np.unique(groups), 1):
            test = groups == subject; train = ~test; train_indices = np.flatnonzero(train)
            assert set(groups[train]).isdisjoint(set(groups[test]))
            cap = cfg["evaluation"].get("max_training_epochs_per_subject")
            if cap:
                selected = []
                for train_subject in np.unique(groups[train_indices]):
                    subject_indices = train_indices[groups[train_indices] == train_subject]
                    selected.extend(np.concatenate([subject_indices[y[subject_indices] == label][:max(1, cap // len(labels))] for label in labels]))
                train_indices = np.asarray(selected)
            inner_groups = groups[train_indices]; inner_splits = min(cfg["evaluation"]["inner_splits"], len(np.unique(inner_groups)))
            search = GridSearchCV(estimator, grid, cv=GroupKFold(inner_splits), scoring="balanced_accuracy", n_jobs=1) if inner_splits >= 2 else estimator
            fitted = search.fit(improved[train_indices], y[train_indices], groups=inner_groups) if inner_splits >= 2 else search.fit(improved[train_indices], y[train_indices])
            fitted = fitted.best_estimator_ if inner_splits >= 2 else fitted
            pred = fitted.predict(improved[test]); fold_score = score(y[test], pred)
            if model_name == "Optimized Random Forest":
                variance_support = fitted.named_steps["variance"].get_support()
                selected_support = fitted.named_steps["select"].get_support()
                selected_indices = np.flatnonzero(variance_support)[np.flatnonzero(selected_support)]
                rf_importance[selected_indices] += fitted.named_steps["model"].feature_importances_
            subject_rows.append({"model": model_name, "subject_id": subject, "fold": fold, **fold_score})
            model_predictions.extend({"subject_id": s, "epoch_id": int(e), "trial_id": int(t), "true_label": int(a), "predicted_label": int(b), "fold": fold, "model": model_name} for e, (s, t, a, b) in enumerate(zip(groups[test], data["trial_id"][test], y[test], pred)))
        predictions.extend(model_predictions); summary = score([p["true_label"] for p in model_predictions], [p["predicted_label"] for p in model_predictions]); summary.update({"Model": model_name, "Experiment": "E2 optimized"}); summaries.append(summary)
    pd.DataFrame({"feature_name": improved_names, "importance": rf_importance / len(np.unique(groups))}).sort_values("importance", ascending=False).to_csv(out / "random_forest_feature_importance.csv", index=False)
    optimized = pd.DataFrame(summaries); comparison = pd.concat([e0, optimized], ignore_index=True); comparison.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False); subjects = pd.DataFrame(subject_rows); subjects.to_csv(out / "per_subject_metrics.csv", index=False)
    subjects.groupby("model")[["Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]].agg(["mean", "std", "median", "min", "max"]).to_csv(out / "subject_metric_statistics.csv")
    class_rows = []
    for model_name in optimized["Model"]:
        subset = [p for p in predictions if p["model"] == model_name]; p, r, f, n = precision_recall_fscore_support([x["true_label"] for x in subset], [x["predicted_label"] for x in subset], labels=labels, zero_division=0)
        class_rows.extend({"model": model_name, "label": int(label), "class_name": name, "precision": a, "recall": b, "f1": c, "support": int(d)} for label, name, a, b, c, d in zip(labels, class_names, p, r, f, n))
    pd.DataFrame(class_rows).to_csv(out / "per_class_metrics.csv", index=False)
    for model_name in optimized["Model"]:
        subset = [p for p in predictions if p["model"] == model_name]; cm = confusion_matrix([p["true_label"] for p in subset], [p["predicted_label"] for p in subset], labels=labels); pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv")
    comparison.set_index("Model")[["Accuracy", "Balanced Accuracy", "Macro F1"]].plot.bar(figsize=(11, 5)); plt.tight_layout(); plt.savefig(out / "model_comparison.png", dpi=150); plt.close()
    best = optimized.sort_values(["Macro F1", "Balanced Accuracy", "Macro Precision"], ascending=False).iloc[0]["Model"]
    subjects[subjects.model == best].set_index("subject_id")[["Accuracy", "Balanced Accuracy", "Macro F1"]].plot.bar(figsize=(12, 5)); plt.tight_layout(); plt.savefig(out / "subject_performance.png", dpi=150); plt.close()
    best_subjects = subjects[subjects.model == best].set_index("subject_id")
    best_subjects[["Accuracy"]].plot.bar(figsize=(12, 4)); plt.tight_layout(); plt.savefig(out / "subject_accuracy.png", dpi=150); plt.close()
    best_subjects[["Macro F1"]].plot.bar(figsize=(12, 4)); plt.tight_layout(); plt.savefig(out / "subject_macro_f1.png", dpi=150); plt.close()
    best_predictions = [p for p in predictions if p["model"] == best]; cm = confusion_matrix([p["true_label"] for p in best_predictions], [p["predicted_label"] for p in best_predictions], labels=labels); plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues"); plt.xticks(range(len(labels)), class_names, rotation=30); plt.yticks(range(len(labels)), class_names); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(out / "confusion_matrix_best_model.png", dpi=150); plt.close()
    print("EEG OPTIMIZATION - FINAL RESULTS\n", comparison.to_string(index=False)); print("Best model:", best, "\nOutput:", out)


if __name__ == "__main__": main()