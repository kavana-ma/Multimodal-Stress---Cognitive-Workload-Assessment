import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


def extract_features(X, sampling_frequency=200, bands=None, nperseg=256):
    bands = bands or {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    if max(high for _, high in bands.values()) >= sampling_frequency / 2:
        raise ValueError("A configured frequency band exceeds the Nyquist frequency")
    names = []
    for channel in range(X.shape[1]):
        names.extend([f"ch{channel + 1}_{name}" for name in ("mean", "std", "variance", "rms", "ptp", "skew", "kurtosis")])
        names.extend(f"ch{channel + 1}_{name}_power" for name in bands)
    statistics = np.stack([X.mean(axis=2), X.std(axis=2), X.var(axis=2), np.sqrt(np.mean(X ** 2, axis=2)),
                           np.ptp(X, axis=2), skew(X, axis=2), kurtosis(X, axis=2)], axis=2)
    frequencies, power = welch(X, fs=sampling_frequency, nperseg=min(nperseg, X.shape[2]), axis=2)
    band_features = []
    for low, high in bands.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_features.append(np.trapezoid(power[:, :, mask], frequencies[mask], axis=2).reshape(X.shape[0], X.shape[1], 1))
    features = np.concatenate([statistics, *band_features], axis=2).reshape(X.shape[0], -1).astype(np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced NaN or infinite values")
    return features, names


def extract_optimized_features(X, channel_names, sampling_frequency=200, bands=None, nperseg=256):
    """Extract interpretable channel-preserving features for the E2 experiment."""
    bands = bands or {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    nyquist = sampling_frequency / 2
    if len(channel_names) != X.shape[1]:
        raise ValueError("Channel names do not match EEG channel dimension")
    if any(high > nyquist or low < 0 for low, high in bands.values()):
        raise ValueError("Configured band is outside the available frequency range")
    frequencies, power = welch(X, fs=sampling_frequency, nperseg=min(nperseg, X.shape[2]), axis=2)
    total_mask = (frequencies >= 1) & (frequencies <= min(45, nyquist))
    total_power = np.trapezoid(power[:, :, total_mask], frequencies[total_mask], axis=2)
    derivative = np.diff(X, axis=2)
    second_derivative = np.diff(derivative, axis=2)
    activity = np.var(X, axis=2)
    mobility = np.sqrt(np.var(derivative, axis=2) / np.maximum(activity, 1e-12))
    derivative_mobility = np.sqrt(np.var(second_derivative, axis=2) / np.maximum(np.var(derivative, axis=2), 1e-12))
    complexity = derivative_mobility / np.maximum(mobility, 1e-12)
    names, blocks = [], []
    for index, channel in enumerate(channel_names):
        prefix = str(channel)
        names.extend([f"{prefix}_{name}" for name in ("mean", "std", "variance", "rms", "ptp", "median", "mad", "skew", "kurtosis", "hjorth_activity", "hjorth_mobility", "hjorth_complexity")])
        blocks.append(np.stack([X[:, index].mean(axis=1), X[:, index].std(axis=1), X[:, index].var(axis=1),
                                np.sqrt(np.mean(X[:, index] ** 2, axis=1)), np.ptp(X[:, index], axis=1),
                                np.median(X[:, index], axis=1), np.median(np.abs(X[:, index] - np.median(X[:, index], axis=1, keepdims=True)), axis=1),
                                skew(X[:, index], axis=1), kurtosis(X[:, index], axis=1), activity[:, index], mobility[:, index], complexity[:, index]], axis=1))
    band_values = {}
    for band_name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        band_values[band_name] = np.trapezoid(power[:, :, mask], frequencies[mask], axis=2)
    for index, channel in enumerate(channel_names):
        for band_name in bands:
            names.extend([f"{channel}_{band_name}_absolute_power", f"{channel}_{band_name}_relative_power"])
        blocks.append(np.stack([value[:, index] for band_name, value in band_values.items() for _ in (0, 1)], axis=1))
        blocks[-1][:, 1::2] = np.stack([value[:, index] / np.maximum(total_power[:, index], 1e-12) for value in band_values.values()], axis=1)
        theta, alpha, beta = (band_values[name][:, index] for name in ("theta", "alpha", "beta"))
        blocks.append(np.stack([theta / np.maximum(alpha, 1e-12), theta / np.maximum(beta, 1e-12), alpha / np.maximum(beta, 1e-12)], axis=1))
        names.extend([f"{channel}_theta_alpha_ratio", f"{channel}_theta_beta_ratio", f"{channel}_alpha_beta_ratio"])
    features = np.concatenate(blocks, axis=1).astype(np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Optimized feature extraction produced NaN or infinite values")
    return features, names