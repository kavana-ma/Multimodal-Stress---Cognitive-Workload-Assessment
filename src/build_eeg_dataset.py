import scipy.io
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("results")

FS = 200

LOWCUT = 1.0
HIGHCUT = 45.0
FILTER_ORDER = 3

TASK_DURATION_SEC = 42
TASK_SAMPLES = FS * TASK_DURATION_SEC


# ============================================================
# FILTER
# ============================================================

sos = butter(
    FILTER_ORDER,
    [LOWCUT, HIGHCUT],
    btype="bandpass",
    fs=FS,
    output="sos"
)


# ============================================================
# STORAGE
# ============================================================

all_segments = []
all_labels = []
all_subjects = []


# ============================================================
# PROCESS SUBJECTS
# ============================================================

subject_dirs = sorted(DATASET_DIR.glob("VP*-EEG"))

print("=" * 70)
print("BUILDING EEG DATASET")
print("=" * 70)

print("Subjects found:", len(subject_dirs))
print()


for subject_dir in subject_dirs:

    subject = subject_dir.name.replace("-EEG", "")

    eeg_file = subject_dir / "cnt_nback.mat"
    marker_file = subject_dir / "mrk_nback.mat"

    if not eeg_file.exists() or not marker_file.exists():
        print(f"{subject}: missing files -> skipped")
        continue


    # --------------------------------------------------------
    # LOAD EEG
    # --------------------------------------------------------

    eeg_data = scipy.io.loadmat(
        eeg_file,
        struct_as_record=False,
        squeeze_me=True
    )

    cnt = eeg_data["cnt_nback"]

    eeg = cnt.x.astype(np.float64)


    # --------------------------------------------------------
    # LOAD MARKERS
    # --------------------------------------------------------

    marker_data = scipy.io.loadmat(
        marker_file,
        struct_as_record=False,
        squeeze_me=True
    )

    mrk = marker_data["mrk_nback"]


    subject_segments = 0


    # --------------------------------------------------------
    # FIND 0-BACK AND 3-BACK SESSIONS
    # --------------------------------------------------------

    for i in range(len(mrk.time)):

        active_classes = []

        for class_index in range(len(mrk.className)):

            if mrk.y[class_index, i] == 1:
                active_classes.append(
                    mrk.className[class_index]
                )


        # ----------------------------------------------------
        # SELECT ONLY 0-BACK AND 3-BACK
        # ----------------------------------------------------

        if "0-back session" in active_classes:

            label = 0

        elif "3-back session" in active_classes:

            label = 1

        else:

            # Ignore 2-back and stimulus markers
            continue


        # ----------------------------------------------------
        # CONVERT MARKER TIME TO SAMPLE
        # ----------------------------------------------------

        start_sample = int(
            mrk.time[i] * FS / 1000
        )

        end_sample = start_sample + TASK_SAMPLES


        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if end_sample > len(eeg):

            print(
                f"{subject}: segment exceeds EEG length -> skipped"
            )

            continue


        # ----------------------------------------------------
        # EXTRACT SEGMENT
        # ----------------------------------------------------

        segment = eeg[
            start_sample:end_sample,
            :
        ]


        # ----------------------------------------------------
        # BAND-PASS FILTER
        # ----------------------------------------------------

        filtered = sosfiltfilt(
            sos,
            segment,
            axis=0
        )


        # ----------------------------------------------------
        # CHANNEL-WISE NORMALIZATION
        # ----------------------------------------------------

        mean = np.mean(
            filtered,
            axis=0
        )

        std = np.std(
            filtered,
            axis=0
        )

        normalized = (
            filtered - mean
        ) / (std + 1e-8)


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        all_segments.append(normalized)
        all_labels.append(label)
        all_subjects.append(subject)

        subject_segments += 1


    print(
        f"{subject}: "
        f"{subject_segments} segments"
    )


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

X = np.stack(all_segments)

y = np.array(
    all_labels,
    dtype=np.int64
)

subjects = np.array(
    all_subjects
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)

print()
print("0-back:", np.sum(y == 0))
print("3-back:", np.sum(y == 1))

print()
print(
    "Unique subjects:",
    len(np.unique(subjects))
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = OUTPUT_DIR / "eeg_0vs3_dataset.npz"

np.savez_compressed(
    output_file,
    X=X,
    y=y,
    subjects=subjects
)

print()
print("Saved to:")
print(output_file)