import numpy as np
import scipy.io
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

print("=" * 70)
print("RAW EEG")
print("=" * 70)

print("Shape:", eeg.shape)
print("Sampling frequency:", cnt.fs)
print("Channels:", eeg.shape[1])


# --------------------------------------------------
# GET FIRST 3-BACK SEGMENT
# --------------------------------------------------

marker_data = scipy.io.loadmat(
    MRK_FILE,
    struct_as_record=False,
    squeeze_me=True
)

mrk = marker_data["mrk_nback"]

first_3back_sample = None

for i in range(len(mrk.time)):

    active_classes = []

    for class_index in range(len(mrk.className)):
        if mrk.y[class_index, i] == 1:
            active_classes.append(mrk.className[class_index])

    if "3-back session" in active_classes:
        first_3back_sample = int(mrk.time[i] * FS / 1000)
        break


start = first_3back_sample
end = start + TASK_SAMPLES

segment = eeg[start:end, :]

print()
print("=" * 70)
print("RAW EEG SEGMENT")
print("=" * 70)

print("Start sample:", start)
print("End sample:", end)
print("Shape:", segment.shape)


# --------------------------------------------------
# DESIGN BUTTERWORTH FILTER
# --------------------------------------------------

sos = butter(
    FILTER_ORDER,
    [LOWCUT, HIGHCUT],
    btype="bandpass",
    fs=FS,
    output="sos"
)


# --------------------------------------------------
# APPLY FILTER
# --------------------------------------------------

filtered_segment = sosfiltfilt(
    sos,
    segment,
    axis=0
)


# --------------------------------------------------
# CHECK RESULT
# --------------------------------------------------

print()
print("=" * 70)
print("FILTERED EEG")
print("=" * 70)

print("Filter:", f"{LOWCUT}-{HIGHCUT} Hz")
print("Order:", FILTER_ORDER)
print("Filtered shape:", filtered_segment.shape)

print()
print("First 5 samples of channel 1 BEFORE filtering:")
print(segment[:5, 0])

print()
print("First 5 samples of channel 1 AFTER filtering:")
print(filtered_segment[:5, 0])

print()
print("Any NaN values:", np.isnan(filtered_segment).any())
print("Any infinite values:", np.isinf(filtered_segment).any())