from typing import Optional, Any, Dict
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """PyTorch dataset for EEG epochs.

    Expects inputs with shape (N_epochs, N_channels, N_samples).
    Each item returns a dict with keys: `x`, `y`, `subject_id`, `trial_id`.

    Tensor convention: channels x time (i.e., (C, T)).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, subject_id: np.ndarray, trial_id: np.ndarray, transform: Optional[Any] = None):
        assert X.ndim == 3, "X must be 3-dimensional (N_epochs, N_channels, N_samples)"
        n_epochs = X.shape[0]
        assert len(y) == n_epochs
        assert len(subject_id) == n_epochs
        assert len(trial_id) == n_epochs

        self.X = X.astype(np.float32)
        self.y = y
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.transform = transform

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.X[idx]  # shape: (C, T)
        y = self.y[idx]
        sid = self.subject_id[idx]
        tid = self.trial_id[idx]

        if self.transform is not None:
            x = self.transform(x)

        # convert to torch tensor when returning
        return {
            "x": torch.from_numpy(x.copy()),
            "y": int(y),
            "subject_id": sid,
            "trial_id": tid,
        }
