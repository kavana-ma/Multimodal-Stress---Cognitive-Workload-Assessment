import argparse

from src.fusion.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run binary workload EEG, fNIRS, and early-fusion LOSO experiments.")
    parser.add_argument("--config", default="configs/fusion.yaml")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()