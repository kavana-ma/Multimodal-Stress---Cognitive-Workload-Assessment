import scipy.io as sio
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt


# ============================================================
# Configuration
# ============================================================

SUBJECTS = [
    "VP001",
    "VP002",
    "VP003",
    "VP004",
    "VP005",
    "VP006",
    "VP007",
    "VP008",
    "VP009",
    "VP010",
    "VP011",
    "VP014",
    "VP015",
    "VP016",
    "VP017",
    "VP018",
    "VP019",
    "VP020",
    "VP021",
    "VP022",
    "VP023",
    "VP024",
    "VP025",
    "VP026",
]

FS = 10.0

LOWCUT = 0.01
HIGHCUT = 0.20
FILTER_ORDER = 3

BASELINE_SECONDS = 5
TASK_SECONDS = 42

BASELINE_SAMPLES = int(BASELINE_SECONDS * FS)
TASK_SAMPLES = int(TASK_SECONDS * FS)


# ============================================================
# Output containers
# ============================================================

X = []
y = []
subjects = []


# ============================================================
# Filter
# ============================================================

sos = butter(
    FILTER_ORDER,
    [LOWCUT, HIGHCUT],
    btype="bandpass",
    fs=FS,
    output="sos"
)


# ============================================================
# Process each subject
# ============================================================

for subject in SUBJECTS:

    print(f"\nProcessing {subject}...")

    cnt_path = os.path.join(
        "dataset",
        f"{subject}-NIRS",
        "cnt_nback.mat"
    )

    mrk_path = os.path.join(
        "dataset",
        f"{subject}-NIRS",
        "mrk_nback.mat"
    )

    # --------------------------------------------------------
    # Load continuous data
    # --------------------------------------------------------

    cnt_data = sio.loadmat(cnt_path)
    cnt = cnt_data["cnt_nback"][0, 0]

    oxy = cnt["oxy"][0, 0]
    deoxy = cnt["deoxy"][0, 0]

    hbo = oxy["x"]
    hbr = deoxy["x"]

    fs = float(oxy["fs"][0, 0])

    if fs != FS:
        raise ValueError(
            f"{subject}: unexpected sampling rate {fs}"
        )

    # --------------------------------------------------------
    # Filter continuous HbO/HbR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load markers
    # --------------------------------------------------------

    mrk_data = sio.loadmat(mrk_path)
    mrk = mrk_data["mrk_nback"][0, 0]

    times_ms = mrk["time"].flatten()
    marker_y = mrk["y"]

    # --------------------------------------------------------
    # Extract sessions
    # --------------------------------------------------------

    subject_count = 0

    for i in range(len(times_ms)):

        class_id = int(
            np.argmax(marker_y[:, i])
        )

        # 0-back -> label 0
        # 2-back -> ignore
        # 3-back -> label 1

        if class_id == 0:
            label = 0

        elif class_id == 2:
            label = 1

        else:
            continue

        marker_sample = int(
            times_ms[i] / 1000 * FS
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_start = (
            marker_sample - BASELINE_SAMPLES
        )

        baseline_end = marker_sample

        # ----------------------------------------------------
        # Task
        # ----------------------------------------------------

        task_start = marker_sample

        task_end = (
            marker_sample + TASK_SAMPLES
        )

        # Safety check
        if baseline_start < 0:
            print(
                f"WARNING: {subject} session {i+1} "
                f"has insufficient baseline"
            )
            continue

        if task_end > hbo_filtered.shape[0]:
            print(
                f"WARNING: {subject} session {i+1} "
                f"has insufficient task data"
            )
            continue

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Baseline correction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Combine HbO + HbR
        # ----------------------------------------------------

        epoch = np.concatenate(
            [
                hbo_corrected,
                hbr_corrected
            ],
            axis=1
        )

        # Expected: 420 × 72
        if epoch.shape != (420, 72):
            raise ValueError(
                f"{subject}: unexpected epoch shape "
                f"{epoch.shape}"
            )

        if not np.isfinite(epoch).all():
            raise ValueError(
                f"{subject}: NaN/Inf detected"
            )

        X.append(epoch.astype(np.float32))
        y.append(label)
        subjects.append(subject)

        subject_count += 1

    print(
        f"  Added {subject_count} epochs"
    )


# ============================================================
# Convert to arrays
# ============================================================

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)
subjects = np.asarray(subjects)


# ============================================================
# Diagnostics
# ============================================================

print("\n" + "=" * 70)
print("FINAL fNIRS DATASET")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)

print("\nLabels:")
print("0-back:", np.sum(y == 0))
print("3-back:", np.sum(y == 1))

print("\nUnique subjects:")
print(len(np.unique(subjects)))

print("\nMean:", X.mean())
print("Std :", X.std())

print("NaN:", np.isnan(X).any())
print("Inf:", np.isinf(X).any())


# ============================================================
# Save
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

output_path = (
    "results/fnirs_0vs3_dataset.npz"
)

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    subjects=subjects
)

print("\nSaved:")
print(output_path)