import argparse
import os

import yaml

from src.fusion.alignment import align_modalities
from src.fusion.data import apply_binary_workload_labels, load_fnirs_provenance, load_modality_data


def main():
    parser = argparse.ArgumentParser(description="Validate and align EEG/fNIRS trials by subject and trial IDs.")
    parser.add_argument("--config", default="configs/fusion.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    load_fnirs_provenance(config)
    _, eeg, eeg_report = load_modality_data(config["data"]["eeg_dir"], "EEG")
    _, fnirs, fnirs_report = load_modality_data(config["data"]["fnirs_dir"], "fNIRS")
    eeg = apply_binary_workload_labels(eeg, config["labels"], "EEG", report_eeg["label_map"])
    fnirs = apply_binary_workload_labels(fnirs, config["labels"], "fNIRS", report_fnirs["label_map"])
    aligned, report = align_modalities(
        eeg[["subject_id", "trial_id", "binary_label"]].rename(columns={"binary_label": "label"}),
        fnirs[["subject_id", "trial_id", "binary_label"]].rename(columns={"binary_label": "label"}),
    )
    output_dir = config["outputs"]["fusion"]
    os.makedirs(output_dir, exist_ok=True)
    aligned.to_csv(os.path.join(output_dir, "aligned_metadata.csv"), index=False)
    print("ALIGNMENT_REPORT", report)
    print("PAIRED_SUBJECTS", aligned["subject_id"].nunique())


if __name__ == "__main__":
    main()