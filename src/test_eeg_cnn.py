import torch
from eeg_cnn import EEGCNN


print("=" * 70)
print("EEG CNN TEST")
print("=" * 70)

model = EEGCNN()

print("Model created successfully.")
print()

# Dummy batch:
# 4 EEG segments
# 8400 time points
# 30 channels

x = torch.randn(4, 8400, 30)

print("Input shape:", x.shape)

output, embedding = model(x)

print("Output shape:", output.shape)
print("Embedding shape:", embedding.shape)