from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


def extract_fnirs_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Summarize channel-wise HbO/HbR epochs without fitting across subjects."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or X.shape[2] < 2:
        raise ValueError(f"Expected fNIRS epochs shaped (N, channels, time>=2), got {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("fNIRS epochs contain NaN or Inf values.")

    n_epochs, n_channels, n_timepoints = X.shape
    time = np.linspace(0.0, 1.0, n_timepoints)
    names = [f"ch_{channel}_{feature}" for channel in range(n_channels) for feature in ("mean", "std", "slope", "peak", "auc")]
    features = np.empty((n_epochs, len(names)), dtype=np.float64)
    for epoch_idx, epoch in enumerate(X):
        for channel_idx, signal in enumerate(epoch):
            offset = channel_idx * 5
            features[epoch_idx, offset:offset + 5] = (
                np.mean(signal),
                np.std(signal),
                np.polyfit(time, signal, 1)[0],
                np.max(np.abs(signal)),
                trapezoid(signal, time),
            )
    return features, names