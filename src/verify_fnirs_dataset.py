import numpy as np


# ============================================================
# Load dataset
# ============================================================

data = np.load(
    "results/fnirs_0vs3_dataset.npz"
)

X = data["X"]
y = data["y"]
subjects = data["subjects"]


# ============================================================
# Basic information
# ============================================================

print("X shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)

print("\nUnique labels:", np.unique(y))
print("Unique subjects:", len(np.unique(subjects)))


# ============================================================
# Check every subject
# ============================================================

print("\nPer-subject counts")
print("=" * 60)

all_ok = True

for subject in np.unique(subjects):

    mask = subjects == subject

    n_total = np.sum(mask)
    n_0 = np.sum(y[mask] == 0)
    n_3 = np.sum(y[mask] == 1)

    ok = (
        n_total == 18
        and n_0 == 9
        and n_3 == 9
    )

    if not ok:
        all_ok = False

    status = "OK" if ok else "CHECK"

    print(
        f"{subject}: "
        f"{status} | "
        f"total={n_total} | "
        f"0-back={n_0} | "
        f"3-back={n_3}"
    )


# ============================================================
# Check subject ordering
# ============================================================

print("\nFirst 20 samples")
print("=" * 60)

for i in range(min(20, len(y))):

    print(
        f"{i:3d}: "
        f"{subjects[i]} | "
        f"label={y[i]}"
    )


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 60)

if all_ok:
    print("DATASET VERIFICATION PASSED.")
else:
    print("DATASET VERIFICATION FAILED.")

print("=" * 60)