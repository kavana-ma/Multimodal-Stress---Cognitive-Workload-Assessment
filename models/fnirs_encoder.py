import torch
import torch.nn as nn


class FNIRSEncoder(nn.Module):
    """
    fNIRS encoder.

    Input:
        (B, 36, 420, 2)

    36  = fNIRS channels
    420 = 42 seconds × 10 Hz
    2   = [HbO, HbR]

    Internally:
        36 channels × 2 chromophores
        -> 72 temporal input channels

    Output:
        features = (B, 128)
        logits   = (B, 3)
    """

    def __init__(
        self,
        n_channels=36,
        n_chromophores=2,
        n_classes=3,
        feature_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_chromophores = n_chromophores
        self.feature_dim = feature_dim

        input_channels = (
            n_channels * n_chromophores
        )

        # ----------------------------------------------------
        # Temporal feature extraction
        # ----------------------------------------------------

        self.temporal = nn.Sequential(

            nn.Conv1d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=7,
                stride=1,
                padding=3,
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=7,
                stride=1,
                padding=3,
            ),

            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Global temporal aggregation
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------
        # 128-D representation
        # ----------------------------------------------------

        self.feature_layer = nn.Sequential(

            nn.Linear(
                256,
                feature_dim
            ),

            nn.ReLU(),

            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            feature_dim,
            n_classes
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(self, x):

        # Input:
        # (B, 36, 420, 2)

        batch_size = x.shape[0]

        # Move chromophore dimension into channels:
        #
        # (B, 36, 420, 2)
        #       ↓
        # (B, 36, 2, 420)
        #       ↓
        # (B, 72, 420)

        x = x.permute(
            0, 1, 3, 2
        )

        x = x.reshape(
            batch_size,
            self.n_channels * self.n_chromophores,
            x.shape[-1]
        )

        x = self.temporal(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        features = self.feature_layer(x)

        return features

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        features = self.extract_features(x)

        logits = self.classifier(features)

        return {
            "features": features,
            "logits": logits,
        }