import scipy.io
import numpy as np
import mne
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from mne.preprocessing import ICA


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

channel_names = [str(ch) for ch in cnt.clab]


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

filtered_segment = sosfiltfilt(
    sos,
    segment,
    axis=0
)


# --------------------------------------------------
# CREATE MNE OBJECT
# --------------------------------------------------

info = mne.create_info(
    ch_names=channel_names,
    sfreq=FS,
    ch_types=["eeg"] * 28 + ["eog", "eog"]
)

raw = mne.io.RawArray(
    filtered_segment.T,
    info
)

print("=" * 70)
print("FILTERED EEG SEGMENT")
print("=" * 70)
print("Shape:", filtered_segment.shape)
print("Duration:", raw.times[-1], "seconds")


# --------------------------------------------------
# ICA
# --------------------------------------------------

ica = ICA(
    n_components=0.99,
    method="fastica",
    random_state=42,
    max_iter=500
)

print()
print("=" * 70)
print("FITTING ICA")
print("=" * 70)

ica.fit(raw)

print()
print("=" * 70)
print("ICA RESULT")
print("=" * 70)

print("Number of ICA components:", ica.n_components_)


# --------------------------------------------------
# FIND EOG-RELATED COMPONENTS
# --------------------------------------------------

print()
print("=" * 70)
print("EOG-RELATED COMPONENTS")
print("=" * 70)

eog_indices, eog_scores = ica.find_bads_eog(
    raw,
    ch_name="VEOG"
)

print("VEOG-related components:", eog_indices)

eog_indices_heog, eog_scores_heog = ica.find_bads_eog(
    raw,
    ch_name="HEOG"
)

print("HEOG-related components:", eog_indices_heog)