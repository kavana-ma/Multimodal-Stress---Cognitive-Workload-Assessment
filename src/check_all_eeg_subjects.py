from pathlib import Path
import scipy.io

dataset_dir = Path("dataset")

print("=" * 70)
print("CHECKING EEG N-BACK SESSIONS FOR ALL SUBJECTS")
print("=" * 70)

for subject_dir in sorted(dataset_dir.glob("VP*-EEG")):

    file_path = subject_dir / "mrk_nback.mat"

    if not file_path.exists():
        print(f"{subject_dir.name}: MISSING")
        continue

    data = scipy.io.loadmat(
        file_path,
        struct_as_record=False,
        squeeze_me=True
    )

    mrk = data["mrk_nback"]

    counts = {
        "0-back": 0,
        "2-back": 0,
        "3-back": 0
    }

    for i in range(len(mrk.time)):

        active_classes = []

        for class_index in range(len(mrk.className)):
            if mrk.y[class_index, i] == 1:
                active_classes.append(mrk.className[class_index])

        for cls in counts:
            if f"{cls} session" in active_classes:
                counts[cls] += 1

    print(
        f"{subject_dir.name}: "
        f"0-back={counts['0-back']}, "
        f"2-back={counts['2-back']}, "
        f"3-back={counts['3-back']}"
    )