import torch
import torch.nn as nn


class FNIRSCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # CNN feature extractor
        # --------------------------------------------------

        self.features = nn.Sequential(

            nn.Conv1d(
                72,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2)
        )

        # --------------------------------------------------
        # LSTM
        #
        # CNN output:
        # [batch, 128, time]
        #
        # LSTM input:
        # [batch, time, 128]
        # --------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # --------------------------------------------------
        # 128-D embedding
        # --------------------------------------------------

        self.embedding = nn.Sequential(

            nn.Linear(128, 128),

            nn.ELU(),

            nn.Dropout(0.3)
        )

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        self.classifier = nn.Linear(
            128,
            2
        )


    def forward(self, x):

        # Input:
        # [batch, 420, 72]

        x = x.transpose(1, 2)

        # [batch, 72, 420]

        x = self.features(x)

        # [batch, 128, ~52]

        # Convert to LSTM format
        x = x.transpose(1, 2)

        # [batch, ~52, 128]

        # LSTM
        lstm_output, (hidden, cell) = self.lstm(x)

        # Take final hidden state
        x = hidden[-1]

        # [batch, 128]

        embedding = self.embedding(x)

        # Classification
        output = self.classifier(embedding)

        return output, embedding