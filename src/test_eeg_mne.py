import scipy.io
import mne
from pathlib import Path


# --------------------------------------------------
# LOAD EEG
# --------------------------------------------------

file_path = Path("dataset/VP001-EEG/cnt_nback.mat")

data = scipy.io.loadmat(
    file_path,
    struct_as_record=False,
    squeeze_me=True
)

cnt = data["cnt_nback"]

eeg = cnt.x
channel_names = [str(ch) for ch in cnt.clab]

print("=" * 70)
print("EEG INFORMATION")
print("=" * 70)

print("EEG shape:", eeg.shape)
print("Sampling frequency:", cnt.fs)
print("Number of channels:", len(channel_names))

print()
print("Channel names:")
print(channel_names)


# --------------------------------------------------
# CREATE MNE RAW OBJECT
# --------------------------------------------------

info = mne.create_info(
    ch_names=channel_names,
    sfreq=cnt.fs,
    ch_types="eeg"
)

raw = mne.io.RawArray(
    eeg.T,
    info
)

print()
print("=" * 70)
print("MNE RAW OBJECT")
print("=" * 70)

print(raw)
print("Number of channels:", len(raw.ch_names))
print("Sampling frequency:", raw.info["sfreq"])
print("Number of samples:", raw.n_times)