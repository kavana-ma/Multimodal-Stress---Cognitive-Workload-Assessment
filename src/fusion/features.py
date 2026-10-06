from __future__ import annotations

from typing import Sequence

import numpy as np


def extract_eeg_features(
    X: np.ndarray,
    channel_names: Sequence[str],
    sampling_frequency: float,
    bands: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Use the repository EEG feature API, with compatibility for the older checkout."""
    try:
        from src.eeg.features import extract_optimized_features
    except ImportError:
        from src.eeg.preprocessing.feature_extraction import extract_feature_matrix

        return extract_feature_matrix(
            np.asarray(X).transpose(0, 2, 1),
            channels=channel_names,
            fs=sampling_frequency,
            bands=bands,
        )

    features, _ = extract_optimized_features(
        np.asarray(X),
        channel_names=channel_names,
        sampling_frequency=sampling_frequency,
        bands=bands,
    )
    return features