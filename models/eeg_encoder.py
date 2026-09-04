import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    """
    EEG encoder for:
        input  = (B, 28, 8400)

    28  = EEG channels
    8400 = 42 seconds × 200 Hz

    Output:
        features = (B, feature_dim)
        logits   = (B, 3)
    """

    def __init__(
        self,
        n_channels=28,
        n_classes=3,
        feature_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # ----------------------------------------------------
        # Temporal feature extraction
        # ----------------------------------------------------

        self.temporal = nn.Sequential(

            nn.Conv1d(
                in_channels=n_channels,
                out_channels=64,
                kernel_size=15,
                stride=2,
                padding=7,
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=15,
                stride=2,
                padding=7,
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=15,
                stride=2,
                padding=7,
            ),

            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Global temporal aggregation
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------
        # Representation layer
        # ----------------------------------------------------

        self.feature_layer = nn.Sequential(

            nn.Linear(256, feature_dim),

            nn.ReLU(),

            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            feature_dim,
            n_classes,
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(self, x):

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