import numpy as np
from typing import Optional, Dict


class LeakSafeScaler:
    """Leakage-safe per-channel z-score scaler.

    Fit on training data only. X shape: (N_epochs, N_channels, N_samples)
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        # compute mean/std per channel across epochs and time
        # reshape to (N_epochs * N_samples, N_channels) by transposing
        n_epochs, n_channels, n_samples = X.shape
        flat = X.transpose(1, 0, 2).reshape(n_channels, -1)
        self.mean_ = flat.mean(axis=1)
        self.std_ = flat.std(axis=1)
        # prevent zero division
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted")
        # apply per-channel
        X_out = (X - self.mean_[None, :, None]) / self.std_[None, :, None]
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
