from pathlib import Path
import sys

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_encoder import EEGEncoder


CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "eeg_encoder_best.pt"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "epochs"
    / "VP001_nback_epochs.mat"
)


def main():

    print()
    print("=" * 70)
    print("EEG CHECKPOINT VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Saved feature dimension: "
        f"{checkpoint['feature_dim']}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{checkpoint['best_val_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Recreate model
    # --------------------------------------------------------

    model = EEGEncoder(
        n_channels=checkpoint["n_channels"],
        n_classes=checkpoint["n_classes"],
        feature_dim=checkpoint["feature_dim"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Load one real EEG epoch
    # --------------------------------------------------------

    print()
    print("Loading real EEG epoch...")

    with h5py.File(DATA_FILE, "r") as f:

        eeg = np.array(
            f["eeg_epochs"]
        )

    # MATLAB v7.3 → Python ordering
    eeg = np.transpose(
        eeg,
        (2, 1, 0)
    )

    # Use first epoch
    x = eeg[0]

    # Shape:
    # (28, 8400)

    x = x.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Apply saved training normalization
    # --------------------------------------------------------

    mean = checkpoint[
        "normalization_mean"
    ]

    std = checkpoint[
        "normalization_std"
    ]

    x = (
        (x - mean.reshape(28, 1))
        / std.reshape(28, 1)
    )

    # Add batch dimension
    x = torch.from_numpy(
        x
    ).unsqueeze(0)

    print(
        f"Input shape: {x.shape}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(x)

    features = output["features"]
    logits = output["logits"]

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = torch.argmax(
        logits,
        dim=1
    )

    print()
    print(
        f"Feature shape: "
        f"{features.shape}"
    )

    print(
        f"Logits shape: "
        f"{logits.shape}"
    )

    print(
        f"Probabilities: "
        f"{probabilities.numpy()}"
    )

    print(
        f"Predicted class: "
        f"{prediction.item()}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert features.shape == (
        1,
        128
    )

    assert logits.shape == (
        1,
        3
    )

    assert probabilities.shape == (
        1,
        3
    )

    assert np.isfinite(
        features.numpy()
    ).all()

    assert np.isfinite(
        logits.numpy()
    ).all()

    print()
    print("✓ Checkpoint loads correctly")
    print("✓ Real EEG epoch accepted")
    print("✓ 128-D representation produced")
    print("✓ 3-class logits produced")
    print("✓ No NaN/Inf in output")

    print()
    print("EEG CHECKPOINT VALIDATION PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    main()