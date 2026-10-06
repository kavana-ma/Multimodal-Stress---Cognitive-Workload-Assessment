import torch
from fnirs_cnn import FNIRSCNN


model = FNIRSCNN()

x = torch.randn(
    4,
    420,
    72
)

output, embedding = model(x)

print("Input:", x.shape)
print("Output:", output.shape)
print("Embedding:", embedding.shape)