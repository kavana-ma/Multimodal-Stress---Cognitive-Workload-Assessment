import numpy as np

# Load EEG
eeg = np.load("results/eeg_0vs3_dataset.npz")

eeg_y = eeg["y"]
eeg_subjects = eeg["subjects"]

# Load fNIRS
fnirs = np.load("results/fnirs_0vs3_dataset.npz")

fnirs_y = fnirs["y"]
fnirs_subjects = fnirs["subjects"]

print("=" * 60)
print("FUSION ALIGNMENT CHECK")
print("=" * 60)

print("\nEEG:")
print("X shape:", eeg["X"].shape)
print("y shape:", eeg_y.shape)
print("subjects shape:", eeg_subjects.shape)

print("\nfNIRS:")
print("X shape:", fnirs["X"].shape)
print("y shape:", fnirs_y.shape)
print("subjects shape:", fnirs_subjects.shape)

# Basic shape check
assert len(eeg_y) == len(fnirs_y)
assert len(eeg_subjects) == len(fnirs_subjects)

print("\n✓ Same number of samples:", len(eeg_y))

# Check sample-by-sample alignment
subject_match = np.all(eeg_subjects == fnirs_subjects)
label_match = np.all(eeg_y == fnirs_y)

print("\nSubject alignment:", subject_match)
print("Label alignment:", label_match)

# Show first 20 samples
print("\nFirst 20 samples:")
print("-" * 60)

for i in range(min(20, len(eeg_y))):
    print(
        f"{i:3d} | "
        f"EEG: {eeg_subjects[i]} label={eeg_y[i]} | "
        f"fNIRS: {fnirs_subjects[i]} label={fnirs_y[i]}"
    )

# Subject-wise verification
print("\n" + "=" * 60)
print("SUBJECT-WISE CHECK")
print("=" * 60)

all_ok = True

for subject in np.unique(eeg_subjects):

    eeg_mask = eeg_subjects == subject
    fnirs_mask = fnirs_subjects == subject

    eeg_labels = eeg_y[eeg_mask]
    fnirs_labels = fnirs_y[fnirs_mask]

    same = np.array_equal(eeg_labels, fnirs_labels)

    print(
        f"{subject}: "
        f"EEG={len(eeg_labels)} samples, "
        f"fNIRS={len(fnirs_labels)} samples, "
        f"labels_match={same}"
    )

    if not same:
        all_ok = False

print("\n" + "=" * 60)

if subject_match and label_match and all_ok:
    print("✓ FUSION ALIGNMENT VERIFIED")
    print("✓ EEG and fNIRS samples can be paired directly.")
else:
    print("✗ ALIGNMENT PROBLEM FOUND")
    print("Do NOT proceed to fusion yet.")

print("=" * 60)