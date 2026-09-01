from src.eeg.evaluation.metrics import compute_classification_metrics
import numpy as np


def test_metrics_basic():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    report = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in report
    assert "f1_macro" in report
