import torch
import torch.nn as nn


class EEGCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=30,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(128, 2)

    def forward(self, x):

        # (batch, time, channels)
        x = x.transpose(1, 2)

        # (batch, 30, 8400)
        x = self.features(x)

        # (batch, 128, 1050)
        x = self.global_pool(x)

        # (batch, 128, 1)
        x = x.squeeze(-1)

        # (batch, 128)
        embedding = self.embedding(x)

        # (batch, 2)
        output = self.classifier(embedding)

        return output, embedding