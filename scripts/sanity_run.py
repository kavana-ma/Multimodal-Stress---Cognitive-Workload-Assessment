from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eeg.data.loader import load_eeg_dataset
from src.eeg.features import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run():
    data = load_eeg_dataset("data/preprocessed/eeg")
    X, y = data["X"], data["y"]
    print("MATLAB VARIABLES:", data["mat_variables"])
    print("SHAPES:", data["raw_X_shape"], "->", X.shape, y.shape)
    print("SUBJECTS:", len(np.unique(data["subject_id"])), "LABELS:", np.unique(y, return_counts=True))
    print("FINITE:", bool(np.isfinite(X).all()), "METADATA:", len(data["metadata"]))
    smoke_indices = np.concatenate([np.flatnonzero(y == label)[:4] for label in np.unique(y)])
    features, names = extract_features(X[smoke_indices], data["preprocessing_config"]["sampling_rate_hz"])
    assert features.shape == (len(smoke_indices), len(names)) and np.isfinite(features).all()
    for model in (make_pipeline(StandardScaler(), SVC()), RandomForestClassifier(n_estimators=10, random_state=42)):
        model.fit(features, y[smoke_indices])
        model.predict(features)
    print("FEATURES:", features.shape, "SANITY_OK")


if __name__ == "__main__":
    run()
