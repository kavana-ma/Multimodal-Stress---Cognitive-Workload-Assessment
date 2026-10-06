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

FS = 200
LOWCUT = 1.0
HIGHCUT = 45.0
FILTER_ORDER = 3


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
# BUTTERWORTH FILTER
# --------------------------------------------------

sos = butter(
    FILTER_ORDER,
    [LOWCUT, HIGHCUT],
    btype="bandpass",
    fs=FS,
    output="sos"
)

filtered_eeg = sosfiltfilt(
    sos,
    eeg,
    axis=0
)


# --------------------------------------------------
# CREATE MNE RAW
# --------------------------------------------------

info = mne.create_info(
    ch_names=channel_names,
    sfreq=FS,
    ch_types=["eeg"] * 28 + ["eog", "eog"]
)

raw = mne.io.RawArray(
    filtered_eeg.T,
    info
)

print("=" * 70)
print("FULL FILTERED EEG")
print("=" * 70)

print("Shape:", filtered_eeg.shape)
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
print("FITTING ICA ON FULL RECORDING")
print("=" * 70)

ica.fit(raw)


print()
print("=" * 70)
print("ICA RESULT")
print("=" * 70)

print("Number of ICA components:", ica.n_components_)


# --------------------------------------------------
# EOG DETECTION
# --------------------------------------------------

print()
print("=" * 70)
print("EOG-RELATED COMPONENTS")
print("=" * 70)

veog_indices, veog_scores = ica.find_bads_eog(
    raw,
    ch_name="VEOG"
)

print("VEOG-related components:", veog_indices)

heog_indices, heog_scores = ica.find_bads_eog(
    raw,
    ch_name="HEOG"
)

print("HEOG-related components:", heog_indices)


# --------------------------------------------------
# COMBINE COMPONENTS
# --------------------------------------------------

eog_components = sorted(
    set(veog_indices + heog_indices)
)

print()
print("Components selected for removal:", eog_components)