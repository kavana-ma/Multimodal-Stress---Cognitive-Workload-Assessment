from pathlib import Path
import sys
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_encoder import EEGEncoder
from models.fnirs_encoder import FNIRSEncoder
from models.fusion_model import FusionModel


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "epochs"

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUBJECTS = [
    "VP001", "VP002", "VP003", "VP004", "VP005", "VP006",
    "VP007", "VP008", "VP009", "VP010", "VP011",
    "VP014", "VP015", "VP016", "VP017", "VP018",
    "VP019", "VP020", "VP021", "VP022", "VP023",
    "VP024", "VP025", "VP026"
]


# ------------------------------------------------------------
# IMPORTANT:
#
# VP026 was used to select the pretrained EEG/fNIRS
# checkpoints.
#
# Therefore we do NOT use VP026 for fusion validation.
#
# VP025 is the fusion development validation subject.
# ------------------------------------------------------------

FUSION_VAL_SUBJECT = "VP025"

FUSION_TRAIN_SUBJECTS = [
    s
    for s in SUBJECTS
    if s not in {
        "VP025",
        "VP026",
    }
]


EEG_CHECKPOINT = (
    CHECKPOINT_DIR
    / "eeg_encoder_best.pt"
)

FNIRS_CHECKPOINT = (
    CHECKPOINT_DIR
    / "fnirs_encoder_best.pt"
)

FUSION_CHECKPOINT = (
    CHECKPOINT_DIR
    / "fusion_best.pt"
)


SEED = 42

BATCH_SIZE = 16

EPOCHS = 40

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

FUSION_DIM = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD SUBJECT
# ============================================================

def load_subject(subject):

    path = (
        DATA_DIR
        / f"{subject}_nback_epochs.mat"
    )

    with h5py.File(path, "r") as f:

        eeg = np.array(
            f["eeg_epochs"]
        )

        fnirs = np.array(
            f["fnirs_epochs"]
        )

        labels = np.array(
            f["labels"]
        ).reshape(-1)

        trial_id = np.array(
            f["trial_id"]
        ).reshape(-1)

    # --------------------------------------------------------
    # MATLAB v7.3 → Python
    # --------------------------------------------------------

    # EEG:
    # h5py = (8400, 28, 27)
    # Python = (27, 28, 8400)

    eeg = np.transpose(
        eeg,
        (2, 1, 0)
    )

    # fNIRS:
    # h5py = (2, 420, 36, 27)
    # Python = (27, 36, 420, 2)

    fnirs = np.transpose(
        fnirs,
        (3, 2, 1, 0)
    )

    labels = labels.astype(
        np.int64
    )

    trial_id = trial_id.astype(
        np.int64
    )

    return (
        eeg,
        fnirs,
        labels,
        trial_id
    )


# ============================================================
# LOAD MULTIPLE SUBJECTS
# ============================================================

def load_subjects(subject_list):

    eeg_all = []

    fnirs_all = []

    labels_all = []

    subjects_all = []

    trials_all = []

    for subject in subject_list:

        eeg, fnirs, labels, trials = (
            load_subject(subject)
        )

        assert eeg.shape == (
            27,
            28,
            8400
        )

        assert fnirs.shape == (
            27,
            36,
            420,
            2
        )

        assert len(labels) == 27

        assert len(trials) == 27

        eeg_all.append(eeg)

        fnirs_all.append(fnirs)

        labels_all.append(labels)

        subjects_all.extend(
            [subject] * 27
        )

        trials_all.append(trials)

    eeg_all = np.concatenate(
        eeg_all,
        axis=0
    )

    fnirs_all = np.concatenate(
        fnirs_all,
        axis=0
    )

    labels_all = np.concatenate(
        labels_all,
        axis=0
    )

    trials_all = np.concatenate(
        trials_all,
        axis=0
    )

    return (
        eeg_all,
        fnirs_all,
        labels_all,
        subjects_all,
        trials_all
    )


# ============================================================
# NORMALIZATION
# ============================================================

def calculate_eeg_normalization(eeg):

    mean = eeg.mean(
        axis=(0, 2),
        keepdims=True
    )

    std = eeg.std(
        axis=(0, 2),
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )

    return mean, std


def calculate_fnirs_normalization(fnirs):

    mean = fnirs.mean(
        axis=(0, 2),
        keepdims=True
    )

    std = fnirs.std(
        axis=(0, 2),
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )

    return mean, std


def normalize_eeg(
    eeg,
    mean,
    std
):

    return (
        (eeg - mean) / std
    ).astype(
        np.float32
    )


def normalize_fnirs(
    fnirs,
    mean,
    std
):

    return (
        (fnirs - mean) / std
    ).astype(
        np.float32
    )


# ============================================================
# LOAD PRETRAINED ENCODERS
# ============================================================

def load_encoders():

    print()
    print("Loading EEG checkpoint...")

    eeg_checkpoint = torch.load(
        EEG_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"EEG checkpoint Macro-F1: "
        f"{eeg_checkpoint['best_val_macro_f1']:.4f}"
    )

    eeg_encoder = EEGEncoder(
        n_channels=28,
        n_classes=3,
        feature_dim=128,
        dropout=0.3
    )

    eeg_encoder.load_state_dict(
        eeg_checkpoint[
            "model_state_dict"
        ]
    )


    print()
    print("Loading fNIRS checkpoint...")

    fnirs_checkpoint = torch.load(
        FNIRS_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"fNIRS checkpoint Macro-F1: "
        f"{fnirs_checkpoint['best_val_macro_f1']:.4f}"
    )

    fnirs_encoder = FNIRSEncoder(
        n_channels=36,
        n_chromophores=2,
        n_classes=3,
        feature_dim=128,
        dropout=0.3
    )

    fnirs_encoder.load_state_dict(
        fnirs_checkpoint[
            "model_state_dict"
        ]
    )

    return (
        eeg_encoder,
        fnirs_encoder
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    total_samples = 0

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for eeg_x, fnirs_x, y in loader:

            eeg_x = eeg_x.to(
                DEVICE
            )

            fnirs_x = fnirs_x.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )

            output = model(
                eeg_x,
                fnirs_x
            )

            logits = output["logits"]

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item()
                * len(y)
            )

            total_samples += len(y)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    loss = (
        total_loss
        / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return (
        loss,
        accuracy,
        macro_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print()
    print("=" * 70)
    print("EEG + fNIRS FUSION TRAINING")
    print("=" * 70)

    print()
    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Fusion training subjects: "
        f"{len(FUSION_TRAIN_SUBJECTS)}"
    )

    print(
        f"Fusion validation subject: "
        f"{FUSION_VAL_SUBJECT}"
    )

    print(
        "VP026: excluded because it was used "
        "for encoder checkpoint selection"
    )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print()
    print(
        "Loading fusion training subjects..."
    )

    (
        train_eeg,
        train_fnirs,
        train_labels,
        _,
        _
    ) = load_subjects(
        FUSION_TRAIN_SUBJECTS
    )

    print(
        f"Training EEG: "
        f"{train_eeg.shape}"
    )

    print(
        f"Training fNIRS: "
        f"{train_fnirs.shape}"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print()
    print(
        "Loading fusion validation subject..."
    )

    (
        val_eeg,
        val_fnirs,
        val_labels,
        _,
        _
    ) = load_subjects(
        [FUSION_VAL_SUBJECT]
    )

    print(
        f"Validation EEG: "
        f"{val_eeg.shape}"
    )

    print(
        f"Validation fNIRS: "
        f"{val_fnirs.shape}"
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    expected_train = (
        len(FUSION_TRAIN_SUBJECTS)
        * 27
    )

    assert train_eeg.shape == (
        expected_train,
        28,
        8400
    )

    assert train_fnirs.shape == (
        expected_train,
        36,
        420,
        2
    )

    assert val_eeg.shape == (
        27,
        28,
        8400
    )

    assert val_fnirs.shape == (
        27,
        36,
        420,
        2
    )

    assert np.isfinite(
        train_eeg
    ).all()

    assert np.isfinite(
        train_fnirs
    ).all()

    # --------------------------------------------------------
    # Normalization
    #
    # These statistics are calculated ONLY from
    # fusion training subjects.
    # --------------------------------------------------------

    print()
    print(
        "Calculating fusion normalization..."
    )

    eeg_mean, eeg_std = (
        calculate_eeg_normalization(
            train_eeg
        )
    )

    fnirs_mean, fnirs_std = (
        calculate_fnirs_normalization(
            train_fnirs
        )
    )

    train_eeg = normalize_eeg(
        train_eeg,
        eeg_mean,
        eeg_std
    )

    val_eeg = normalize_eeg(
        val_eeg,
        eeg_mean,
        eeg_std
    )

    train_fnirs = normalize_fnirs(
        train_fnirs,
        fnirs_mean,
        fnirs_std
    )

    val_fnirs = normalize_fnirs(
        val_fnirs,
        fnirs_mean,
        fnirs_std
    )

    # --------------------------------------------------------
    # Tensors
    # --------------------------------------------------------

    train_eeg = torch.from_numpy(
        train_eeg
    )

    train_fnirs = torch.from_numpy(
        train_fnirs
    )

    train_labels = torch.from_numpy(
        train_labels
    )

    val_eeg = torch.from_numpy(
        val_eeg
    )

    val_fnirs = torch.from_numpy(
        val_fnirs
    )

    val_labels = torch.from_numpy(
        val_labels
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        train_eeg,
        train_fnirs,
        train_labels
    )

    val_dataset = TensorDataset(
        val_eeg,
        val_fnirs,
        val_labels
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Load pretrained encoders
    # --------------------------------------------------------

    eeg_encoder, fnirs_encoder = (
        load_encoders()
    )

    # --------------------------------------------------------
    # Fusion model
    # --------------------------------------------------------

    model = FusionModel(
        eeg_encoder=eeg_encoder,
        fnirs_encoder=fnirs_encoder,
        eeg_dim=128,
        fnirs_dim=128,
        fusion_dim=FUSION_DIM,
        n_classes=3,
        dropout=0.3,
        freeze_encoders=True
    ).to(DEVICE)

    # --------------------------------------------------------
    # Verify frozen encoders
    # --------------------------------------------------------

    assert all(
        not p.requires_grad
        for p in model.eeg_encoder.parameters()
    )

    assert all(
        not p.requires_grad
        for p in model.fnirs_encoder.parameters()
    )

    # --------------------------------------------------------
    # Optimizer
    #
    # Only trainable parameters:
    #   EEG projection
    #   fNIRS projection
    #   fusion classifier
    # --------------------------------------------------------

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_f1 = -1.0

    print()
    print("-" * 70)
    print("TRAINING")
    print("-" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        total_samples = 0

        for eeg_x, fnirs_x, y in train_loader:

            eeg_x = eeg_x.to(
                DEVICE
            )

            fnirs_x = fnirs_x.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )

            optimizer.zero_grad()

            output = model(
                eeg_x,
                fnirs_x
            )

            logits = output["logits"]

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * len(y)
            )

            total_samples += len(y)

        train_loss = (
            running_loss
            / total_samples
        )

        (
            val_loss,
            val_acc,
            val_f1
        ) = evaluate(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Save best fusion checkpoint
        # ----------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "eeg_dim":
                        128,

                    "fnirs_dim":
                        128,

                    "fusion_dim":
                        FUSION_DIM,

                    "n_classes":
                        3,

                    "dropout":
                        0.3,

                    "freeze_encoders":
                        True,

                    "eeg_checkpoint":
                        str(
                            EEG_CHECKPOINT
                        ),

                    "fnirs_checkpoint":
                        str(
                            FNIRS_CHECKPOINT
                        ),

                    "eeg_normalization_mean":
                        eeg_mean,

                    "eeg_normalization_std":
                        eeg_std,

                    "fnirs_normalization_mean":
                        fnirs_mean,

                    "fnirs_normalization_std":
                        fnirs_std,

                    "fusion_train_subjects":
                        FUSION_TRAIN_SUBJECTS,

                    "fusion_validation_subject":
                        FUSION_VAL_SUBJECT,

                    "excluded_encoder_validation_subject":
                        "VP026",

                    "best_val_macro_f1":
                        best_f1,

                    "seed":
                        SEED,
                },
                FUSION_CHECKPOINT
            )

            print(
                f"  -> Saved best fusion checkpoint "
                f"(Macro-F1={best_f1:.4f})"
            )

    print()
    print("=" * 70)
    print("FUSION TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Macro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Checkpoint saved to:\n"
        f"{FUSION_CHECKPOINT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()