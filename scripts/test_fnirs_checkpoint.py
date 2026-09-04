from pathlib import Path
import sys

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from models.fnirs_encoder import FNIRSEncoder


CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "fnirs_encoder_best.pt"
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
    print("fNIRS CHECKPOINT VALIDATION")
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

    model = FNIRSEncoder(
        n_channels=checkpoint["n_channels"],
        n_chromophores=checkpoint["n_chromophores"],
        n_classes=checkpoint["n_classes"],
        feature_dim=checkpoint["feature_dim"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Load real fNIRS epoch
    # --------------------------------------------------------

    print()
    print("Loading real fNIRS epoch...")

    with h5py.File(DATA_FILE, "r") as f:

        fnirs = np.array(
            f["fnirs_epochs"]
        )

    # MATLAB v7.3 → Python ordering
    #
    # h5py:
    # (2, 420, 36, 27)
    #
    # Python:
    # (27, 36, 420, 2)

    fnirs = np.transpose(
        fnirs,
        (3, 2, 1, 0)
    )

    # First real epoch
    x = fnirs[0]

    x = x.astype(
        np.float32
    )

    print(
        f"Input shape: {x.shape}"
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

    # Remove training-only singleton dimensions.
    # Final shape:
    # (36, 1, 2)

    mean = mean.reshape(
        36, 1, 2
    )

    std = std.reshape(
        36, 1, 2
    )

    x = (
        (x - mean)
        / std
    ).astype(
        np.float32
    )

    # Add batch dimension
    x = torch.from_numpy(
        x
    ).unsqueeze(0)

    print(
        f"Normalized input shape: "
        f"{x.shape}"
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
    print("✓ Real fNIRS epoch accepted")
    print("✓ 128-D representation produced")
    print("✓ 3-class logits produced")
    print("✓ No NaN/Inf in output")

    print()
    print(
        "fNIRS CHECKPOINT VALIDATION PASSED."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()