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


from models.eeg_encoder import EEGEncoder
from models.fnirs_encoder import FNIRSEncoder
from models.fusion_model import FusionModel


def main():

    print()
    print("=" * 70)
    print("FUSION MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create pretrained encoder architectures
    # --------------------------------------------------------

    eeg_encoder = EEGEncoder(
        n_channels=28,
        n_classes=3,
        feature_dim=128,
        dropout=0.3
    )

    fnirs_encoder = FNIRSEncoder(
        n_channels=36,
        n_chromophores=2,
        n_classes=3,
        feature_dim=128,
        dropout=0.3
    )

    # --------------------------------------------------------
    # Create fusion model
    # --------------------------------------------------------

    model = FusionModel(
        eeg_encoder=eeg_encoder,
        fnirs_encoder=fnirs_encoder,
        eeg_dim=128,
        fnirs_dim=128,
        fusion_dim=128,
        n_classes=3,
        dropout=0.3,
        freeze_encoders=True
    )

    model.eval()

    # --------------------------------------------------------
    # Dummy paired batch
    # --------------------------------------------------------

    eeg_x = torch.randn(
        4,
        28,
        8400
    )

    fnirs_x = torch.randn(
        4,
        36,
        420,
        2
    )

    print()
    print(
        f"EEG input shape: "
        f"{eeg_x.shape}"
    )

    print(
        f"fNIRS input shape: "
        f"{fnirs_x.shape}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            eeg_x,
            fnirs_x
        )

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print()
    print(
        f"EEG representation: "
        f"{output['z_eeg'].shape}"
    )

    print(
        f"fNIRS representation: "
        f"{output['z_fnirs'].shape}"
    )

    print(
        f"EEG projection: "
        f"{output['p_eeg'].shape}"
    )

    print(
        f"fNIRS projection: "
        f"{output['p_fnirs'].shape}"
    )

    print(
        f"Fused representation: "
        f"{output['z_fused'].shape}"
    )

    print(
        f"Logits: "
        f"{output['logits'].shape}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert output["z_eeg"].shape == (
        4,
        128
    )

    assert output["z_fnirs"].shape == (
        4,
        128
    )

    assert output["p_eeg"].shape == (
        4,
        128
    )

    assert output["p_fnirs"].shape == (
        4,
        128
    )

    assert output["z_fused"].shape == (
        4,
        256
    )

    assert output["logits"].shape == (
        4,
        3
    )

    # --------------------------------------------------------
    # Verify encoders are actually frozen
    # --------------------------------------------------------

    frozen_eeg = all(
        not param.requires_grad
        for param in model.eeg_encoder.parameters()
    )

    frozen_fnirs = all(
        not param.requires_grad
        for param in model.fnirs_encoder.parameters()
    )

    assert frozen_eeg
    assert frozen_fnirs

    print()
    print(
        "✓ EEG representation = 128-D"
    )

    print(
        "✓ fNIRS representation = 128-D"
    )

    print(
        "✓ EEG projection = 128-D"
    )

    print(
        "✓ fNIRS projection = 128-D"
    )

    print(
        "✓ Concatenated representation = 256-D"
    )

    print(
        "✓ Fusion logits = 3 classes"
    )

    print(
        "✓ EEG encoder frozen"
    )

    print(
        "✓ fNIRS encoder frozen"
    )

    print()
    print(
        "FUSION MODEL TEST PASSED."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()