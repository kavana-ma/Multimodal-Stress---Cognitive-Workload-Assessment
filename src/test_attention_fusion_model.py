import numpy as np
import torch

from attention_fusion_model import AttentionFusionModel


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

eeg_data = np.load(
    "results/eeg_0vs3_dataset.npz"
)

fnirs_data = np.load(
    "results/fnirs_0vs3_dataset.npz"
)

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

model = AttentionFusionModel()


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

output, fused_embedding, eeg_weight, fnirs_weight = model(
    eeg,
    fnirs
)


# --------------------------------------------------
# Print shapes
# --------------------------------------------------

print("=" * 60)
print("ATTENTION FUSION MODEL TEST")
print("=" * 60)

print("EEG input:", eeg.shape)
print("fNIRS input:", fnirs.shape)

print("Output:", output.shape)

print("Fused embedding:", fused_embedding.shape)

print("EEG weight:", eeg_weight.shape)
print("fNIRS weight:", fnirs_weight.shape)

print("\nSample EEG weights:")
print(eeg_weight.squeeze(1).detach().numpy())

print("\nSample fNIRS weights:")
print(fnirs_weight.squeeze(1).detach().numpy())

print("=" * 60)