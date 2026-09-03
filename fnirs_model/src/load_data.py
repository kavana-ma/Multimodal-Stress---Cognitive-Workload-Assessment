import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to .mat file
DATA_PATH = Path("../data/fnirs_all_subjects.mat")

# Open MATLAB v7.3 file
with h5py.File(DATA_PATH, "r") as f:

    X = np.array(f["X_fnirs"])
    y = np.array(f["y"]).squeeze()
    subject_id = np.array(f["subjectID"]).squeeze()
    trial_id = np.array(f["trialID"]).squeeze()

# MATLAB stores dimensions reversed
X = np.transpose(X, (3, 2, 1, 0))

print("=" * 50)
print("TU BERLIN fNIRS DATASET LOADED")
print("=" * 50)

print("X shape        :", X.shape)
print("y shape        :", y.shape)
print("Subjects       :", len(np.unique(subject_id)))

print("\nClass distribution")
print("Low    :", np.sum(y == 0))
print("Medium :", np.sum(y == 1))
print("High   :", np.sum(y == 2))

# Visualize one sample
epoch = 0
channel = 0

plt.figure(figsize=(8,4))
plt.plot(X[epoch, channel, :, 0], label="HbO", color="red")
plt.plot(X[epoch, channel, :, 1], label="HbR", color="blue")
plt.axhline(0, color="black", linestyle="--")
plt.title("Sample fNIRS Epoch")
plt.xlabel("Samples")
plt.ylabel("Normalized concentration")
plt.legend()
plt.tight_layout()
plt.show()