import torch
import torch.nn as nn

from eeg_cnn import EEGCNN
from fnirs_cnn import FNIRSCNN


class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Modality-specific feature extractors
        # --------------------------------------------------

        self.eeg_encoder = EEGCNN()

        self.fnirs_encoder = FNIRSCNN()

        # --------------------------------------------------
        # Fusion classifier
        # EEG embedding   = 128
        # fNIRS embedding = 128
        # Concatenated    = 256
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
        # Extract EEG embedding
        # --------------------------------------------------

        _, eeg_embedding = self.eeg_encoder(eeg)

        # --------------------------------------------------
        # Extract fNIRS embedding
        # --------------------------------------------------

        _, fnirs_embedding = self.fnirs_encoder(fnirs)

        # --------------------------------------------------
        # Concatenate the two modality embeddings
        # --------------------------------------------------

        fused_embedding = torch.cat(
            [eeg_embedding, fnirs_embedding],
            dim=1
        )

        # --------------------------------------------------
        # Final classification
        # --------------------------------------------------

        output = self.fusion_classifier(fused_embedding)

        return output, fused_embedding