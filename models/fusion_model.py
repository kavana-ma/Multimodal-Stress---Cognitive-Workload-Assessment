import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """
    Learned representation-level EEG + fNIRS fusion.

    EEG input:
        (B, 28, 8400)

    fNIRS input:
        (B, 36, 420, 2)

    EEG encoder:
        128-D representation

    fNIRS encoder:
        128-D representation

    Fusion:
        EEG projection
        +
        fNIRS projection
        ↓
        concatenation
        ↓
        fusion classification head

    Output:
        EEG features
        fNIRS features
        projected features
        fused representation
        logits
    """

    def __init__(
        self,
        eeg_encoder,
        fnirs_encoder,
        eeg_dim=128,
        fnirs_dim=128,
        fusion_dim=128,
        n_classes=3,
        dropout=0.3,
        freeze_encoders=True,
    ):
        super().__init__()

        self.eeg_encoder = eeg_encoder
        self.fnirs_encoder = fnirs_encoder

        self.eeg_dim = eeg_dim
        self.fnirs_dim = fnirs_dim
        self.fusion_dim = fusion_dim

        # ----------------------------------------------------
        # Freeze pretrained encoders
        # ----------------------------------------------------

        if freeze_encoders:

            for param in self.eeg_encoder.parameters():
                param.requires_grad = False

            for param in self.fnirs_encoder.parameters():
                param.requires_grad = False

        # ----------------------------------------------------
        # Modality-specific projection layers
        # ----------------------------------------------------

        self.eeg_proj = nn.Linear(
            eeg_dim,
            fusion_dim
        )

        self.fnirs_proj = nn.Linear(
            fnirs_dim,
            fusion_dim
        )

        # ----------------------------------------------------
        # Fusion classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                2 * fusion_dim,
                fusion_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                fusion_dim,
                n_classes
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        eeg_x,
        fnirs_x
    ):

        # ----------------------------------------------------
        # Extract modality representations
        # ----------------------------------------------------

        z_eeg = self.eeg_encoder.extract_features(
            eeg_x
        )

        z_fnirs = self.fnirs_encoder.extract_features(
            fnirs_x
        )

        # ----------------------------------------------------
        # Project each modality
        # ----------------------------------------------------

        p_eeg = self.eeg_proj(
            z_eeg
        )

        p_fnirs = self.fnirs_proj(
            z_fnirs
        )

        # ----------------------------------------------------
        # Representation-level fusion
        # ----------------------------------------------------

        z_fused = torch.cat(
            [
                p_eeg,
                p_fnirs
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            z_fused
        )

        return {
            "z_eeg": z_eeg,
            "z_fnirs": z_fnirs,
            "p_eeg": p_eeg,
            "p_fnirs": p_fnirs,
            "z_fused": z_fused,
            "logits": logits,
        }