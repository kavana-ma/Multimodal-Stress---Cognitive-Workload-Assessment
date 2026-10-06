import numpy as np
import torch

from fusion_model import FusionModel


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

eeg_data = np.load("results/eeg_0vs3_dataset.npz")
fnirs_data = np.load("results/fnirs_0vs3_dataset.npz")

eeg = torch.tensor(
    eeg_data["X"][:4],
    dtype=torch.float32
)

fnirs = torch.tensor(
    fnirs_data["X"][:4],
    dtype=torch.float32
)

# --------------------------------------------------
# Create model
# --------------------------------------------------

model = FusionModel()

# --------------------------------------------------
# Forward pass
# --------------------------------------------------

output, fused_embedding = model(eeg, fnirs)

# --------------------------------------------------
# Print shapes
# --------------------------------------------------

print("=" * 60)
print("FUSION MODEL TEST")
print("=" * 60)

print("EEG input:", eeg.shape)
print("fNIRS input:", fnirs.shape)

print("Output:", output.shape)
print("Fused embedding:", fused_embedding.shape)

print("=" * 60)