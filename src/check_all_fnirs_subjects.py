import scipy.io as sio
import numpy as np
import os


# ============================================================
# Subjects in the fusion dataset
# ============================================================

subjects = [
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


# ============================================================
# Expected values
# ============================================================

EXPECTED_FS = 10.0
EXPECTED_CHANNELS = 36
EXPECTED_MARKERS = 27

all_ok = True


# ============================================================
# Check every subject
# ============================================================

for subject in subjects:

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

    try:

        # -----------------------------
        # Load continuous data
        # -----------------------------

        cnt_data = sio.loadmat(cnt_path)
        cnt = cnt_data["cnt_nback"][0, 0]

        oxy = cnt["oxy"][0, 0]
        deoxy = cnt["deoxy"][0, 0]

        hbo = oxy["x"]
        hbr = deoxy["x"]

        fs = float(oxy["fs"][0, 0])

        # -----------------------------
        # Load markers
        # -----------------------------

        mrk_data = sio.loadmat(mrk_path)
        mrk = mrk_data["mrk_nback"][0, 0]

        times_ms = mrk["time"].flatten()
        y = mrk["y"]

        # -----------------------------
        # Counts
        # -----------------------------

        count_0 = int(np.sum(y[0] == 1))
        count_2 = int(np.sum(y[1] == 1))
        count_3 = int(np.sum(y[2] == 1))

        # -----------------------------
        # Basic checks
        # -----------------------------

        ok = True

        if fs != EXPECTED_FS:
            ok = False

        if hbo.shape[1] != EXPECTED_CHANNELS:
            ok = False

        if hbr.shape[1] != EXPECTED_CHANNELS:
            ok = False

        if len(times_ms) != EXPECTED_MARKERS:
            ok = False

        if count_0 != 9:
            ok = False

        if count_2 != 9:
            ok = False

        if count_3 != 9:
            ok = False

        # Check matching HbO/HbR time dimension
        if hbo.shape[0] != hbr.shape[0]:
            ok = False

        status = "OK" if ok else "CHECK"

        print(
            f"{subject}: "
            f"{status} | "
            f"fs={fs:g} | "
            f"HbO={hbo.shape} | "
            f"HbR={hbr.shape} | "
            f"markers={len(times_ms)} | "
            f"0={count_0} 2={count_2} 3={count_3}"
        )

        if not ok:
            all_ok = False

    except Exception as e:

        all_ok = False

        print(
            f"{subject}: ERROR | {e}"
        )


# ============================================================
# Final result
# ============================================================

print("\n" + "=" * 80)

if all_ok:
    print("ALL 24 SUBJECTS PASSED THE fNIRS STRUCTURE CHECK.")
else:
    print("SOME SUBJECTS NEED ATTENTION.")

print("=" * 80)