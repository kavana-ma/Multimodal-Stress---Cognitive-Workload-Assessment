from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eeg.data.loader import load_eeg_dataset
from src.eeg.features import extract_features


def metrics(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred), "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}


def main():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/eeg_baseline.yaml").read_text(encoding="utf-8"))
    np.random.seed(config["runtime"]["seed"])
    print("=" * 52 + "\nEEG BASELINE PIPELINE\n" + "=" * 52)
    data = load_eeg_dataset(root / config["dataset"]["path"])
    X, y, groups = data["X"], data["y"], data["subject_id"]
    print(f"Loaded {X.shape} from MATLAB {data['raw_X_shape']} (epoch axis transposed)")
    feature_cfg = config["features"]
    features, feature_names = extract_features(X, feature_cfg["sampling_frequency_hz"], feature_cfg["bands"], feature_cfg["welch_nperseg"])
    out = root / config["runtime"]["results_dir"]; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature_name": feature_names}).to_csv(out / "feature_names.csv", index=False)
    pd.DataFrame({"class_key": [f"class_{v}" for v in np.unique(y)], "label": np.unique(y),
                  "class_name": [data["label_map"].get(f"class_{v}", str(v)) for v in np.unique(y)]}).to_csv(out / "label_mapping.csv", index=False)
    print(f"Extracted {features.shape[1]} features; subjects={len(np.unique(groups))}, classes={len(np.unique(y))}")
    models = {
        "SVM": (Pipeline([("scaler", StandardScaler()), ("classifier", SVC(class_weight=config["models"]["svm"]["class_weight"]))]),
                {"classifier__C": config["models"]["svm"]["grid"]["C"], "classifier__gamma": config["models"]["svm"]["grid"]["gamma"]}),
        "Random Forest": (RandomForestClassifier(n_estimators=config["models"]["random_forest"]["n_estimators"], class_weight="balanced", random_state=config["runtime"]["seed"], n_jobs=-1),
                          {"max_depth": config["models"]["random_forest"]["max_depth"]})}
    all_predictions, summaries, per_subject = [], [], []
    subjects = np.unique(groups)
    for model_name, (estimator, grid) in models.items():
        print(f"Running LOSO {model_name} ({len(subjects)} folds)...")
        predictions = []
        for subject in subjects:
            test = groups == subject; train = ~test
            train_indices = np.flatnonzero(train)
            cap = config["evaluation"].get("max_training_epochs_per_subject")
            if cap:
                selected = []
                for train_subject in np.unique(groups[train]):
                    subject_indices = train_indices[groups[train_indices] == train_subject]
                    for label in np.unique(y[subject_indices]):
                        selected.extend(subject_indices[y[subject_indices] == label][:max(1, cap // len(np.unique(y)))])
                train_indices = np.asarray(selected, dtype=int)
            inner_groups = groups[train_indices]
            inner_splits = min(config["evaluation"]["inner_splits"], len(np.unique(inner_groups)))
            if inner_splits >= 2:
                search = GridSearchCV(estimator, grid, cv=GroupKFold(inner_splits), scoring="balanced_accuracy", n_jobs=1)
                search.fit(features[train_indices], y[train_indices], groups=inner_groups); fitted = search.best_estimator_
            else:
                fitted = estimator.fit(features[train_indices], y[train_indices])
            pred = fitted.predict(features[test]); fold = metrics(y[test], pred)
            per_subject.append({"Model": model_name, "Held-out Subject": subject, "Number of Test Epochs": int(test.sum()), **fold})
            predictions.extend({"Subject ID": s, "Trial ID": t, "True Label": int(a), "Predicted Label": int(b), "Model": model_name} for s, t, a, b in zip(groups[test], data["trial_id"][test], y[test], pred))
        all_predictions.extend(predictions); summary = metrics([p["True Label"] for p in predictions], [p["Predicted Label"] for p in predictions]); summary["Model"] = model_name; summaries.append(summary)
        cm = confusion_matrix(y, [p["Predicted Label"] for p in predictions], labels=np.unique(y)); fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax); ax.set_xticks(range(len(np.unique(y))), [data["label_map"].get(f"class_{v}", str(v)) for v in np.unique(y)], rotation=30); ax.set_yticks(range(len(np.unique(y))), [data["label_map"].get(f"class_{v}", str(v)) for v in np.unique(y)]); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(out / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150); plt.close(fig)
    pd.DataFrame(summaries).to_csv(out / "model_comparison.csv", index=False); pd.DataFrame(per_subject).to_csv(out / "per_subject_results.csv", index=False); pd.DataFrame(all_predictions).to_csv(out / "predictions.csv", index=False)
    class_rows = []
    for model_name in models:
        model_predictions = [p for p in all_predictions if p["Model"] == model_name]
        precision, recall, f1, support = precision_recall_fscore_support([p["True Label"] for p in model_predictions], [p["Predicted Label"] for p in model_predictions], labels=np.unique(y), zero_division=0)
        class_rows.extend({"Model": model_name, "Label": int(label), "Class Name": data["label_map"].get(f"class_{label}", str(label)), "Precision": p, "Recall": r, "F1": score, "Support": int(count)} for label, p, r, score, count in zip(np.unique(y), precision, recall, f1, support))
    pd.DataFrame(class_rows).to_csv(out / "per_class_metrics.csv", index=False)
    comparison = pd.DataFrame(summaries).set_index("Model")[["Accuracy", "Balanced Accuracy", "Macro F1"]]; comparison.plot.bar(figsize=(7, 4)); plt.tight_layout(); plt.savefig(out / "model_comparison.png", dpi=150); plt.close()
    pd.DataFrame(per_subject).pivot(index="Held-out Subject", columns="Model", values="Macro F1").plot.bar(figsize=(12, 4)); plt.tight_layout(); plt.savefig(out / "per_subject_macro_f1.png", dpi=150); plt.close()
    print("FINAL RESULTS\n", comparison.to_string(), "\nResults saved to:", out)


if __name__ == "__main__": main()