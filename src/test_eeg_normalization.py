import scipy.io
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

subject = "VP001"

EEG_FILE = Path(f"dataset/{subject}-EEG/cnt_nback.mat")
MRK_FILE = Path(f"dataset/{subject}-EEG/mrk_nback.mat")

FS = 200
LOWCUT = 1.0
HIGHCUT = 45.0
FILTER_ORDER = 3

TASK_DURATION_SEC = 42
TASK_SAMPLES = FS * TASK_DURATION_SEC


# --------------------------------------------------
# LOAD EEG
# --------------------------------------------------

data = scipy.io.loadmat(
    EEG_FILE,
    struct_as_record=False,
    squeeze_me=True
)

cnt = data["cnt_nback"]

eeg = cnt.x.astype(np.float64)


# --------------------------------------------------
# LOAD MARKERS
# --------------------------------------------------

marker_data = scipy.io.loadmat(
    MRK_FILE,
    struct_as_record=False,
    squeeze_me=True
)

mrk = marker_data["mrk_nback"]


# --------------------------------------------------
# FIND FIRST 3-BACK SESSION
# --------------------------------------------------

start_sample = None

for i in range(len(mrk.time)):

    active_classes = []

    for class_index in range(len(mrk.className)):
        if mrk.y[class_index, i] == 1:
            active_classes.append(mrk.className[class_index])

    if "3-back session" in active_classes:
        start_sample = int(mrk.time[i] * FS / 1000)
        break


end_sample = start_sample + TASK_SAMPLES

segment = eeg[start_sample:end_sample, :]


# --------------------------------------------------
# BAND-PASS FILTER
# --------------------------------------------------

sos = butter(
    FILTER_ORDER,
    [LOWCUT, HIGHCUT],
    btype="bandpass",
    fs=FS,
    output="sos"
)

filtered = sosfiltfilt(
    sos,
    segment,
    axis=0
)


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

mean = np.mean(filtered, axis=0)
std = np.std(filtered, axis=0)

normalized = (filtered - mean) / (std + 1e-8)


# --------------------------------------------------
# CHECK RESULTS
# --------------------------------------------------

print("=" * 70)
print("NORMALIZATION CHECK")
print("=" * 70)

print("Original shape:", segment.shape)
print("Normalized shape:", normalized.shape)

print()
print("Channel 1:")
print("Mean before normalization:", mean[0])
print("Std before normalization:", std[0])

print()
print("Channel 1 after normalization:")
print("Mean:", np.mean(normalized[:, 0]))
print("Std:", np.std(normalized[:, 0]))

print()
print("All channels:")
print(
    "Mean range:",
    np.min(np.mean(normalized, axis=0)),
    "to",
    np.max(np.mean(normalized, axis=0))
)

print(
    "Std range:",
    np.min(np.std(normalized, axis=0)),
    "to",
    np.max(np.std(normalized, axis=0))
)

print()
print("Any NaN:", np.isnan(normalized).any())
print("Any infinite:", np.isinf(normalized).any())