from pathlib import Path
import h5py
import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(
    r"D:\major_project_group50\project\FUSION\data\epochs"
)

EXPECTED_SUBJECTS = [
    "VP001", "VP002", "VP003", "VP004", "VP005", "VP006",
    "VP007", "VP008", "VP009", "VP010", "VP011",
    "VP014", "VP015", "VP016", "VP017", "VP018", "VP019",
    "VP020", "VP021", "VP022", "VP023", "VP024", "VP025",
    "VP026"
]

EXPECTED_EEG_SHAPE = (27, 28, 8400)
EXPECTED_FNIRS_SHAPE = (27, 36, 420, 2)

VALID_LABELS = {0, 1, 2}


# ============================================================
# READ MATLAB VARIABLE
# ============================================================

def read_variable(f, name):

    if name not in f:
        raise KeyError(f"Missing variable: {name}")

    return np.array(f[name])


# ============================================================
# CHECK ONE SUBJECT
# ============================================================

def check_subject(subject):

    file_path = DATA_DIR / f"{subject}_nback_epochs.mat"

    if not file_path.exists():

        print(
            f"[FAIL] {subject}: file not found"
        )

        return False

    try:

        with h5py.File(file_path, "r") as f:

            # ------------------------------------------------
            # Required variables
            # ------------------------------------------------

            required = [
                "eeg_epochs",
                "fnirs_epochs",
                "labels",
                "subject_ids",
                "trial_id",
                "eeg_start_times",
                "nirs_start_times",
            ]

            for variable in required:

                if variable not in f:

                    print(
                        f"[FAIL] {subject}: "
                        f"missing '{variable}'"
                    )

                    return False

            # ------------------------------------------------
            # Load data
            # ------------------------------------------------

            eeg = read_variable(
                f, "eeg_epochs"
            )

            fnirs = read_variable(
                f, "fnirs_epochs"
            )

            labels = read_variable(
                f, "labels"
            ).reshape(-1)

            trial_id = read_variable(
                f, "trial_id"
            ).reshape(-1)

            eeg_start = read_variable(
                f, "eeg_start_times"
            ).reshape(-1)

            nirs_start = read_variable(
                f, "nirs_start_times"
            ).reshape(-1)

            # ------------------------------------------------
            # MATLAB v7.3 dimension order
            # ------------------------------------------------
            #
            # MATLAB logical shapes:
            #
            # EEG:
            #   27 x 28 x 8400
            #
            # fNIRS:
            #   27 x 36 x 420 x 2
            #
            # h5py reads MATLAB dimensions reversed:
            #
            # EEG:
            #   8400 x 28 x 27
            #
            # fNIRS:
            #   2 x 420 x 36 x 27
            #
            # Convert to Python convention.
            # ------------------------------------------------

            eeg = np.transpose(
                eeg,
                (2, 1, 0)
            )

            fnirs = np.transpose(
                fnirs,
                (3, 2, 1, 0)
            )

            # ------------------------------------------------
            # Shape checks
            # ------------------------------------------------

            if eeg.shape != EXPECTED_EEG_SHAPE:

                print(
                    f"[FAIL] {subject}: "
                    f"EEG shape = {eeg.shape}, "
                    f"expected = {EXPECTED_EEG_SHAPE}"
                )

                return False

            if fnirs.shape != EXPECTED_FNIRS_SHAPE:

                print(
                    f"[FAIL] {subject}: "
                    f"fNIRS shape = {fnirs.shape}, "
                    f"expected = {EXPECTED_FNIRS_SHAPE}"
                )

                return False

            if labels.shape != (27,):

                print(
                    f"[FAIL] {subject}: "
                    f"labels shape = {labels.shape}"
                )

                return False

            if trial_id.shape != (27,):

                print(
                    f"[FAIL] {subject}: "
                    f"trial_id shape = {trial_id.shape}"
                )

                return False

            if eeg_start.shape != (27,):

                print(
                    f"[FAIL] {subject}: "
                    f"eeg_start_times shape = "
                    f"{eeg_start.shape}"
                )

                return False

            if nirs_start.shape != (27,):

                print(
                    f"[FAIL] {subject}: "
                    f"nirs_start_times shape = "
                    f"{nirs_start.shape}"
                )

                return False

            # ------------------------------------------------
            # NaN / Inf
            # ------------------------------------------------

            if not np.isfinite(eeg).all():

                print(
                    f"[FAIL] {subject}: "
                    f"EEG contains NaN/Inf"
                )

                return False

            if not np.isfinite(fnirs).all():

                print(
                    f"[FAIL] {subject}: "
                    f"fNIRS contains NaN/Inf"
                )

                return False

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            labels_int = labels.astype(int)

            unique_labels = set(
                np.unique(labels_int)
            )

            if unique_labels != VALID_LABELS:

                print(
                    f"[FAIL] {subject}: "
                    f"labels = {sorted(unique_labels)}, "
                    f"expected = {sorted(VALID_LABELS)}"
                )

                return False

            # ------------------------------------------------
            # Trial IDs
            # ------------------------------------------------

            trial_int = trial_id.astype(int)

            expected_trials = np.arange(1, 28)

            if not np.array_equal(
                trial_int,
                expected_trials
            ):

                print(
                    f"[FAIL] {subject}: "
                    f"trial_id is not 1..27"
                )

                return False

            # ------------------------------------------------
            # Start times
            # ------------------------------------------------

            if not np.isfinite(eeg_start).all():

                print(
                    f"[FAIL] {subject}: "
                    f"invalid EEG start times"
                )

                return False

            if not np.isfinite(nirs_start).all():

                print(
                    f"[FAIL] {subject}: "
                    f"invalid fNIRS start times"
                )

                return False

            # ------------------------------------------------
            # Basic EEG statistics
            # ------------------------------------------------

            eeg_min = float(np.min(eeg))
            eeg_max = float(np.max(eeg))

            # ------------------------------------------------
            # Basic fNIRS statistics
            # ------------------------------------------------

            fnirs_min = float(np.min(fnirs))
            fnirs_max = float(np.max(fnirs))

            # ------------------------------------------------
            # Class distribution
            # ------------------------------------------------

            counts = {
                label: int(
                    np.sum(labels_int == label)
                )
                for label in sorted(unique_labels)
            }

            print(
                f"[PASS] {subject} | "
                f"EEG={eeg.shape} | "
                f"fNIRS={fnirs.shape} | "
                f"classes={counts}"
            )

            return True

    except Exception as e:

        print(
            f"[FAIL] {subject}: "
            f"{type(e).__name__}: {e}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FUSION DATASET VALIDATION")
    print("=" * 70)
    print()

    print(
        f"Dataset directory:\n{DATA_DIR}"
    )

    print()

    files = sorted(
        DATA_DIR.glob("*_nback_epochs.mat")
    )

    print(
        f"MAT files found: {len(files)}"
    )

    print()

    passed = 0
    failed = 0

    for subject in EXPECTED_SUBJECTS:

        if check_subject(subject):

            passed += 1

        else:

            failed += 1

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Subjects expected : "
        f"{len(EXPECTED_SUBJECTS)}"
    )

    print(
        f"Subjects passed   : "
        f"{passed}"
    )

    print(
        f"Subjects failed   : "
        f"{failed}"
    )

    print(
        f"Total paired epochs: "
        f"{passed * 27}"
    )

    if failed == 0:

        print()
        print("ALL 24 SUBJECTS PASSED.")
        print()
        print("Total paired epochs = 648")
        print()
        print(
            "DATASET VALIDATION COMPLETE."
        )
        print(
            "Ready for Python model development."
        )

    else:

        print()
        print(
            "DATASET VALIDATION FAILED."
        )
        print(
            "Fix the reported errors before "
            "continuing."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()