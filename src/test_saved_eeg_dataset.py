import numpy as np

DATA_FILE = "results/eeg_0vs3_dataset.npz"

data = np.load(DATA_FILE)

X = data["X"]
y = data["y"]
subjects = data["subjects"]

print("=" * 70)
print("SAVED EEG DATASET CHECK")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)

print()

print("Number of subjects:", len(np.unique(subjects)))

print("0-back samples:", np.sum(y == 0))
print("3-back samples:", np.sum(y == 1))

print()

print("Overall mean:", np.mean(X))
print("Overall std:", np.std(X))

print()

print("Any NaN:", np.isnan(X).any())
print("Any infinite:", np.isinf(X).any())

print()

print("First 10 subject IDs:")
print(subjects[:10])

print()

print("First 10 labels:")
print(y[:10])

print()

print("Unique labels:")
print(np.unique(y))