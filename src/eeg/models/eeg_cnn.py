import torch
import torch.nn as nn


class SimpleEEGCNN(nn.Module):
    """Simple EEG CNN baseline.

    Input: (batch, channels, samples)
    Outputs: embedding (batch, embedding_dim), logits (batch, num_classes)
    """

    def __init__(self, n_channels: int, n_samples: int, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # temporal conv
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: (batch, channels, samples)
        h = self.conv1(x)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = torch.relu(h)
        h = self.pool(h).squeeze(-1)  # (batch, 64)
        emb = self.embedding(h)
        logits = self.classifier(emb)
        return emb, logits
