from pathlib import Path


dataset_dir = Path("dataset")

eeg_subjects = {
    p.name.replace("-EEG", "")
    for p in dataset_dir.glob("VP*-EEG")
}

fnirs_subjects = {
    p.name.replace("-NIRS", "")
    for p in dataset_dir.glob("VP*-NIRS")
}


common_subjects = sorted(
    eeg_subjects & fnirs_subjects
)

eeg_only = sorted(
    eeg_subjects - fnirs_subjects
)

fnirs_only = sorted(
    fnirs_subjects - eeg_subjects
)


print("=" * 70)
print("FUSION SUBJECT CHECK")
print("=" * 70)

print()
print("EEG subjects:", len(eeg_subjects))
print("fNIRS subjects:", len(fnirs_subjects))
print("Common subjects:", len(common_subjects))

print()
print("Common subjects:")
print(common_subjects)

print()

print("EEG-only subjects:")
print(eeg_only)

print()

print("fNIRS-only subjects:")
print(fnirs_only)