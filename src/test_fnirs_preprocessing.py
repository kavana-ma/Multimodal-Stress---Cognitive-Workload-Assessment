import scipy.io as sio
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

subject = "VP001-NIRS"

LOWCUT = 0.01
HIGHCUT = 0.20
FILTER_ORDER = 3

BASELINE_SECONDS = 5
TASK_SECONDS = 42


# ============================================================
# Load data
# ============================================================

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

cnt_data = sio.loadmat(cnt_path)
mrk_data = sio.loadmat(mrk_path)

cnt = cnt_data["cnt_nback"][0, 0]

oxy = cnt["oxy"][0, 0]
deoxy = cnt["deoxy"][0, 0]

hbo = oxy["x"]
hbr = deoxy["x"]

fs = float(oxy["fs"][0, 0])

mrk = mrk_data["mrk_nback"][0, 0]

times_ms = mrk["time"].flatten()
y = mrk["y"]


# ============================================================
# Filter continuous recording
# ============================================================

sos = butter(
    FILTER_ORDER,
    [LOWCUT, HIGHCUT],
    btype="bandpass",
    fs=fs,
    output="sos"
)

hbo_filtered = sosfiltfilt(
    sos,
    hbo,
    axis=0
)

hbr_filtered = sosfiltfilt(
    sos,
    hbr,
    axis=0
)


# ============================================================
# Select first session
# ============================================================

session_index = 0

marker_sample = int(
    times_ms[session_index] / 1000 * fs
)

class_id = np.argmax(
    y[:, session_index]
)

class_names = [
    "0-back",
    "2-back",
    "3-back"
]

print("Subject:", subject)
print("Session:", session_index + 1)
print("Class:", class_names[class_id])
print("Marker sample:", marker_sample)


# ============================================================
# Extract baseline and task
# ============================================================

baseline_samples = int(
    BASELINE_SECONDS * fs
)

task_samples = int(
    TASK_SECONDS * fs
)

baseline_start = marker_sample - baseline_samples
baseline_end = marker_sample

task_start = marker_sample
task_end = marker_sample + task_samples


hbo_baseline = hbo_filtered[
    baseline_start:baseline_end
]

hbr_baseline = hbr_filtered[
    baseline_start:baseline_end
]

hbo_task = hbo_filtered[
    task_start:task_end
]

hbr_task = hbr_filtered[
    task_start:task_end
]


# ============================================================
# Baseline correction
# ============================================================

hbo_baseline_mean = np.mean(
    hbo_baseline,
    axis=0
)

hbr_baseline_mean = np.mean(
    hbr_baseline,
    axis=0
)

hbo_corrected = (
    hbo_task - hbo_baseline_mean
)

hbr_corrected = (
    hbr_task - hbr_baseline_mean
)


# ============================================================
# Combine HbO + HbR
# ============================================================

fnirs_epoch = np.concatenate(
    [
        hbo_corrected,
        hbr_corrected
    ],
    axis=1
)


# ============================================================
# Print diagnostics
# ============================================================

print("\nShapes")
print("-" * 50)

print("HbO corrected:", hbo_corrected.shape)
print("HbR corrected:", hbr_corrected.shape)
print("Combined:", fnirs_epoch.shape)

print("\nExpected:")
print("(420, 72)")

print("\nStatistics")
print("-" * 50)

print(
    "Mean:",
    np.mean(fnirs_epoch)
)

print(
    "Std:",
    np.std(fnirs_epoch)
)

print(
    "Min:",
    np.min(fnirs_epoch)
)

print(
    "Max:",
    np.max(fnirs_epoch)
)

print("\nNaN:", np.isnan(fnirs_epoch).any())
print("Inf:", np.isinf(fnirs_epoch).any())


# ============================================================
# Plot corrected HbO
# ============================================================

time = np.arange(task_samples) / fs

plt.figure(figsize=(14, 7))

for ch in range(6):

    plt.plot(
        time,
        hbo_corrected[:, ch],
        label=f"HbO Channel {ch + 1}"
    )

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Baseline-corrected HbO")

plt.title(
    f"{subject} — "
    f"{class_names[class_id]} — "
    f"Preprocessed 42s Epoch"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()