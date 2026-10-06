import scipy.io as sio
import numpy as np
import os

subject = "VP001-NIRS"

cnt_path = os.path.join(
    "dataset",
    subject,
    "cnt_nback.mat"
)

mrk_path = os.path.join(
    "dataset",
    subject,
    "mrk_nback.mat"
)

# -----------------------------
# Load data
# -----------------------------

cnt_data = sio.loadmat(cnt_path)
mrk_data = sio.loadmat(mrk_path)

cnt = cnt_data["cnt_nback"][0, 0]

oxy = cnt["oxy"][0, 0]
deoxy = cnt["deoxy"][0, 0]

oxy_signal = oxy["x"]
deoxy_signal = deoxy["x"]

fs = float(oxy["fs"][0, 0])

# -----------------------------
# Load markers
# -----------------------------

mrk = mrk_data["mrk_nback"][0, 0]

times_ms = mrk["time"].flatten()
y = mrk["y"]

# -----------------------------
# First session
# -----------------------------

marker_sample = int(times_ms[0] / 1000 * fs)

baseline_start = marker_sample - int(5 * fs)
baseline_end = marker_sample

task_start = marker_sample
task_end = marker_sample + int(42 * fs)

oxy_baseline = oxy_signal[
    baseline_start:baseline_end
]

oxy_task = oxy_signal[
    task_start:task_end
]

deoxy_baseline = deoxy_signal[
    baseline_start:baseline_end
]

deoxy_task = deoxy_signal[
    task_start:task_end
]

# -----------------------------
# Baseline statistics
# -----------------------------

oxy_baseline_mean = np.mean(
    oxy_baseline,
    axis=0
)

deoxy_baseline_mean = np.mean(
    deoxy_baseline,
    axis=0
)

print("Baseline correction check")
print("=" * 60)

print("\nHbO baseline means:")
print(oxy_baseline_mean)

print("\nHbR baseline means:")
print(deoxy_baseline_mean)

# -----------------------------
# Apply baseline correction
# -----------------------------

oxy_corrected = (
    oxy_task - oxy_baseline_mean
)

deoxy_corrected = (
    deoxy_task - deoxy_baseline_mean
)

print("\nAfter baseline correction")
print("=" * 60)

print("\nHbO:")
print(
    "Mean across channels:",
    np.mean(oxy_corrected)
)

print(
    "Std:",
    np.std(oxy_corrected)
)

print("\nHbR:")
print(
    "Mean across channels:",
    np.mean(deoxy_corrected)
)

print(
    "Std:",
    np.std(deoxy_corrected)
)

print("\nShape:")
print("HbO:", oxy_corrected.shape)
print("HbR:", deoxy_corrected.shape)