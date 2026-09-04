import os
import sys
import h5py
import torch
import numpy as np

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.eeg_encoder import EEGEncoder
from models.fnirs_encoder import FNIRSEncoder
from models.fusion_model import FusionModel


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EPOCH_FILE = os.path.join(
    BASE_DIR, "data", "epochs", "VP001_nback_epochs.mat"
)

FUSION_CKPT = os.path.join(
    BASE_DIR, "models", "checkpoints", "fusion_best.pt"
)

EEG_CKPT = os.path.join(
    BASE_DIR, "models", "checkpoints", "eeg_encoder_best.pt"
)

FNIRS_CKPT = os.path.join(
    BASE_DIR, "models", "checkpoints", "fnirs_encoder_best.pt"
)


print("=" * 60)
print("FUSION CHECKPOINT VALIDATION")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


# ---------------------------------------------------------
# Load fusion checkpoint
# ---------------------------------------------------------
print("\nLoading fusion checkpoint...")

fusion_ckpt = torch.load(
    FUSION_CKPT,
    map_location=device,
    weights_only=False
)

print("Fusion checkpoint loaded.")

if "best_val_f1" in fusion_ckpt:
    print(f"Best validation Macro-F1: {fusion_ckpt['best_val_f1']:.4f}")


# ---------------------------------------------------------
# Load encoder checkpoints
# ---------------------------------------------------------
print("\nLoading EEG checkpoint...")

eeg_ckpt = torch.load(
    EEG_CKPT,
    map_location=device,
    weights_only=False
)

print("Loading fNIRS checkpoint...")

fnirs_ckpt = torch.load(
    FNIRS_CKPT,
    map_location=device,
    weights_only=False
)


# ---------------------------------------------------------
# Build models
# ---------------------------------------------------------
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

eeg_encoder.load_state_dict(
    eeg_ckpt["model_state_dict"]
)

fnirs_encoder.load_state_dict(
    fnirs_ckpt["model_state_dict"]
)

fusion_model = FusionModel(
    eeg_encoder=eeg_encoder,
    fnirs_encoder=fnirs_encoder,
    eeg_dim=128,
    fnirs_dim=128,
    fusion_dim=128,
    n_classes=3,
    dropout=0.3,
    freeze_encoders=True
)

fusion_model.load_state_dict(
    fusion_ckpt["model_state_dict"]
)

fusion_model = fusion_model.to(device)
fusion_model.eval()


# ---------------------------------------------------------
# Load real epoch
# ---------------------------------------------------------
print("\nLoading real paired epoch...")

with h5py.File(EPOCH_FILE, "r") as f:

    eeg = np.array(f["eeg_epochs"], dtype=np.float32)
    fnirs = np.array(f["fnirs_epochs"], dtype=np.float32)
    labels = np.array(f["labels"]).reshape(-1)

# MATLAB/HDF5 orientation
# EEG: (8400, 28, 27) -> (27, 28, 8400)
eeg = np.transpose(eeg, (2, 1, 0))

# fNIRS: (2, 420, 36, 27) -> (27, 36, 420, 2)
fnirs = np.transpose(fnirs, (3, 2, 1, 0))


# ---------------------------------------------------------
# Select first epoch
# ---------------------------------------------------------
eeg_x = eeg[0:1]
fnirs_x = fnirs[0:1]
y_true = int(labels[0])


print(f"EEG input:   {eeg_x.shape}")
print(f"fNIRS input: {fnirs_x.shape}")
print(f"True label:  {y_true}")


# ---------------------------------------------------------
# Apply fusion normalization
# ---------------------------------------------------------
print("\nApplying fusion normalization...")

if "eeg_mean" in fusion_ckpt:
    eeg_mean = np.asarray(
        fusion_ckpt["eeg_mean"],
        dtype=np.float32
    ).reshape(1, 28, 1)

    eeg_std = np.asarray(
        fusion_ckpt["eeg_std"],
        dtype=np.float32
    ).reshape(1, 28, 1)

    eeg_x = (eeg_x - eeg_mean) / (eeg_std + 1e-8)


if "fnirs_mean" in fusion_ckpt:
    fnirs_mean = np.asarray(
        fusion_ckpt["fnirs_mean"],
        dtype=np.float32
    ).reshape(1, 36, 1, 2)

    fnirs_std = np.asarray(
        fusion_ckpt["fnirs_std"],
        dtype=np.float32
    ).reshape(1, 36, 1, 2)

    fnirs_x = (
        fnirs_x - fnirs_mean
    ) / (fnirs_std + 1e-8)


# ---------------------------------------------------------
# Convert to tensors
# ---------------------------------------------------------
eeg_tensor = torch.from_numpy(eeg_x).to(device)
fnirs_tensor = torch.from_numpy(fnirs_x).to(device)


# ---------------------------------------------------------
# Forward pass
# ---------------------------------------------------------
print("\nRunning fusion model...")

with torch.no_grad():

    output = fusion_model(
        eeg_tensor,
        fnirs_tensor
    )

    logits = output["logits"]

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"EEG representation:   {output['z_eeg'].shape}")
print(f"fNIRS representation: {output['z_fnirs'].shape}")
print(f"EEG projection:        {output['p_eeg'].shape}")
print(f"fNIRS projection:      {output['p_fnirs'].shape}")
print(f"Fused representation:  {output['z_fused'].shape}")
print(f"Logits:                {logits.shape}")

print("\nProbabilities:")
print(probabilities.cpu().numpy())

print(f"\nTrue class:      {y_true}")
print(f"Predicted class: {prediction.item()}")

print(
    f"\nCorrect: "
    f"{prediction.item() == y_true}"
)


# ---------------------------------------------------------
# Numerical sanity checks
# ---------------------------------------------------------
assert not torch.isnan(logits).any()
assert not torch.isinf(logits).any()

assert not torch.isnan(probabilities).any()
assert not torch.isinf(probabilities).any()

assert output["z_eeg"].shape == (1, 128)
assert output["z_fnirs"].shape == (1, 128)
assert output["z_fused"].shape == (1, 256)
assert logits.shape == (1, 3)

print("\nNo NaN/Inf detected.")
print("All expected dimensions passed.")

print("\n" + "=" * 60)
print("FUSION CHECKPOINT VALIDATION PASSED")
print("=" * 60)