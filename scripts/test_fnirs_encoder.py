from pathlib import Path
import sys

import torch


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from models.fnirs_encoder import FNIRSEncoder


def main():

    print()
    print("=" * 60)
    print("fNIRS ENCODER TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = FNIRSEncoder(
        n_channels=36,
        n_chromophores=2,
        n_classes=3,
        feature_dim=128,
        dropout=0.3,
    )

    model.eval()

    # --------------------------------------------------------
    # Fake fNIRS batch
    # --------------------------------------------------------

    x = torch.randn(
        4,
        36,
        420,
        2
    )

    print()
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

    print(
        f"Feature shape: {features.shape}"
    )

    print(
        f"Logits shape: {logits.shape}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert features.shape == (
        4,
        128
    )

    assert logits.shape == (
        4,
        3
    )

    print()
    print("✓ Input shape correct")
    print("✓ Feature shape correct")
    print("✓ Logits shape correct")
    print()
    print("fNIRS ENCODER TEST PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()