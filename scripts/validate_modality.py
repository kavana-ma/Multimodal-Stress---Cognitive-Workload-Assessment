import argparse
import json

import yaml

from src.fusion.data import apply_binary_workload_labels, load_fnirs_provenance, load_modality_data


def main():
    parser = argparse.ArgumentParser(description="Validate one processed workload modality.")
    parser.add_argument("--modality", choices=("eeg", "fnirs"), required=True)
    parser.add_argument("--config", default="configs/fusion.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    data_dir = config["data"][f"{args.modality}_dir"]
    X, metadata, report = load_modality_data(data_dir, args.modality.upper())
    if args.modality == "fnirs":
        _, provenance = load_fnirs_provenance(config)
        representation = provenance["representation"]
        report["representation"] = representation
        report["preprocessing_provenance"] = provenance
    binary = apply_binary_workload_labels(metadata, config["labels"], args.modality.upper(), report["label_map"])
    report["binary_distribution"] = {str(k): int(v) for k, v in binary["binary_label"].value_counts().sort_index().items()}
    print(json.dumps(report, indent=2))
    print(f"VALIDATION_OK: {X.shape[0]} {args.modality.upper()} trials")


if __name__ == "__main__":
    main()