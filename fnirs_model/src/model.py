"""
04_model.py
Lightweight CNN-LSTM for fNIRS
"""

import torch
import torch.nn as nn

class FNIRSCNNLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(72, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 3)

        # NEW: Initialize weights
        self._init_weights()

    # NEW
    def _init_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        # (B,300,72) -> (B,72,300)
        x = x.permute(0, 2, 1)

        x = self.cnn(x)

        # (B,32,150)
        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(x)

        x = hidden[-1]
        x = self.dropout(x)

        return self.fc(x)


if __name__ == "__main__":

    model = FNIRSCNNLSTM()

    sample = torch.randn(32, 300, 72)

    out = model(sample)

    print("Input :", sample.shape)
    print("Output:", out.shape)