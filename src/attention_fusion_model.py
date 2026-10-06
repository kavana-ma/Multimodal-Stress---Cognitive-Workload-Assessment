import torch
import torch.nn as nn

from eeg_cnn import EEGCNN
from fnirs_cnn import FNIRSCNN


class AttentionFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Modality-specific encoders
        # --------------------------------------------------

        self.eeg_encoder = EEGCNN()
        self.fnirs_encoder = FNIRSCNN()

        # --------------------------------------------------
        # Learnable modality gates
        #
        # Each gate receives the 128-D embedding and
        # produces a scalar importance value.
        # --------------------------------------------------

        self.eeg_gate = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fnirs_gate = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # --------------------------------------------------
        # Fusion classifier
        #
        # Weighted EEG       = 128-D
        # Weighted fNIRS     = 128-D
        # Concatenated       = 256-D
        # --------------------------------------------------

        self.fusion_classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ELU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ELU(),

            nn.Dropout(0.3),

            nn.Linear(64, 2)
        )

    def forward(self, eeg, fnirs):

        # --------------------------------------------------
        # Extract modality embeddings
        # --------------------------------------------------

        _, eeg_embedding = self.eeg_encoder(eeg)

        _, fnirs_embedding = self.fnirs_encoder(fnirs)

        # --------------------------------------------------
        # Calculate modality importance
        # --------------------------------------------------

        eeg_weight = self.eeg_gate(
            eeg_embedding
        )

        fnirs_weight = self.fnirs_gate(
            fnirs_embedding
        )

        # --------------------------------------------------
        # Apply learned weights
        #
        # [batch, 1] × [batch, 128]
        #       ↓
        # [batch, 128]
        # --------------------------------------------------

        weighted_eeg = (
            eeg_embedding * eeg_weight
        )

        weighted_fnirs = (
            fnirs_embedding * fnirs_weight
        )

        # --------------------------------------------------
        # Fuse weighted embeddings
        # --------------------------------------------------

        fused_embedding = torch.cat(
            [
                weighted_eeg,
                weighted_fnirs
            ],
            dim=1
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        output = self.fusion_classifier(
            fused_embedding
        )

        return (
            output,
            fused_embedding,
            eeg_weight,
            fnirs_weight
        )