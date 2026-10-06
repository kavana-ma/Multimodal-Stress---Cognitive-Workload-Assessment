import scipy.io
import numpy as np
from pathlib import Path

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

subject = "VP001"

EEG_FILE = Path(f"dataset/{subject}-EEG/cnt_nback.mat")
MRK_FILE = Path(f"dataset/{subject}-EEG/mrk_nback.mat")

FS = 200
TASK_DURATION_SEC = 42

TASK_SAMPLES = FS * TASK_DURATION_SEC


# --------------------------------------------------
# LOAD EEG
# --------------------------------------------------

eeg_data = scipy.io.loadmat(
    EEG_FILE,
    struct_as_record=False,
    squeeze_me=True
)

cnt = eeg_data["cnt_nback"]

eeg = cnt.x

print("=" * 70)
print("EEG DATA")
print("=" * 70)
print("Subject:", subject)
print("EEG shape:", eeg.shape)
print("Sampling frequency:", cnt.fs)
print("Channels:", eeg.shape[1])


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
# FIND 0-BACK AND 3-BACK SESSIONS
# --------------------------------------------------

segments = []

for i in range(len(mrk.time)):

    active_classes = []

    for class_index in range(len(mrk.className)):
        if mrk.y[class_index, i] == 1:
            active_classes.append(mrk.className[class_index])

    session = None
    label = None

    if "0-back session" in active_classes:
        session = "0-back"
        label = 0

    elif "3-back session" in active_classes:
        session = "3-back"
        label = 1

    else:
        continue

    # Marker time is in milliseconds
    start_sample = int(mrk.time[i] * FS / 1000)

    end_sample = start_sample + TASK_SAMPLES

    segment = eeg[start_sample:end_sample, :]

    print(
        f"{session:8s} | "
        f"start sample = {start_sample:7d} | "
        f"end sample = {end_sample:7d} | "
        f"shape = {segment.shape}"
    )

    segments.append((segment, label))


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("Total extracted segments:", len(segments))

zero_back = sum(label == 0 for _, label in segments)
three_back = sum(label == 1 for _, label in segments)

print("0-back segments:", zero_back)
print("3-back segments:", three_back)

if segments:
    X = np.stack([segment for segment, _ in segments])
    y = np.array([label for _, label in segments])

    print("Final X shape:", X.shape)
    print("Final y shape:", y.shape)