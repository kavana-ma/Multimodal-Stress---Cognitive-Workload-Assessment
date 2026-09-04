import sys
from pathlib import Path

import torch


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_encoder import EEGEncoder


def main():

    print()
    print("=" * 60)
    print("EEG ENCODER TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = EEGEncoder(
        n_channels=28,
        n_classes=3,
        feature_dim=128,
        dropout=0.3,
    )

    model.eval()

    # --------------------------------------------------------
    # Fake batch
    # --------------------------------------------------------

    x = torch.randn(
        4,
        28,
        8400,
    )

    print()
    print(f"Input shape: {x.shape}")

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
        128,
    )

    assert logits.shape == (
        4,
        3,
    )

    print()
    print("✓ Input shape correct")
    print("✓ Feature shape correct")
    print("✓ Logits shape correct")
    print()
    print("EEG ENCODER TEST PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()