import numpy as np
import torch
from src.eeg.models.eeg_cnn import SimpleEEGCNN


def test_cnn_forward():
    batch = 4
    C = 8
    T = 100
    x = torch.randn(batch, C, T)
    model = SimpleEEGCNN(n_channels=C, n_samples=T, embedding_dim=32, num_classes=3)
    emb, logits = model(x)
    assert emb.shape == (batch, 32)
    assert logits.shape == (batch, 3)
