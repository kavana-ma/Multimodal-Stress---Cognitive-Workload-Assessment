from pathlib import Path
import sys
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------------------
# Project import
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_encoder import EEGEncoder


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "epochs"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

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

# One subject is held out for validation.
# This is ONLY for encoder development.
VAL_SUBJECT = "VP026"

TRAIN_SUBJECTS = [
    s for s in SUBJECTS
    if s != VAL_SUBJECT
]

SEED = 42

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
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
# LOAD EEG FROM MATLAB V7.3
# ============================================================

def load_eeg_subject(subject):

    path = DATA_DIR / f"{subject}_nback_epochs.mat"

    with h5py.File(path, "r") as f:

        eeg = np.array(
            f["eeg_epochs"]
        )

        labels = np.array(
            f["labels"]
        ).reshape(-1)

        trial_id = np.array(
            f["trial_id"]
        ).reshape(-1)

    # MATLAB v7.3 dimension order:
    #
    # h5py:
    #   (8400, 28, 27)
    #
    # Python:
    #   (27, 28, 8400)

    eeg = np.transpose(
        eeg,
        (2, 1, 0)
    )

    labels = labels.astype(
        np.int64
    )

    trial_id = trial_id.astype(
        np.int64
    )

    return eeg, labels, trial_id


# ============================================================
# LOAD MULTIPLE SUBJECTS
# ============================================================

def load_subjects(subject_list):

    eeg_list = []
    label_list = []
    subject_list_out = []
    trial_list = []

    for subject in subject_list:

        eeg, labels, trial_id = (
            load_eeg_subject(subject)
        )

        eeg_list.append(eeg)
        label_list.append(labels)

        subject_list_out.extend(
            [subject] * len(labels)
        )

        trial_list.append(trial_id)

    eeg = np.concatenate(
        eeg_list,
        axis=0
    )

    labels = np.concatenate(
        label_list,
        axis=0
    )

    trials = np.concatenate(
        trial_list,
        axis=0
    )

    return (
        eeg,
        labels,
        subject_list_out,
        trials
    )


# ============================================================
# TRAINING-ONLY NORMALIZATION
# ============================================================

def calculate_normalization(eeg):

    """
    Calculate channel-wise mean/std using
    TRAINING DATA ONLY.

    Input:
        (N, 28, 8400)

    Statistics:
        one mean/std per EEG channel.
    """

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


def normalize(eeg, mean, std):

    return (
        (eeg - mean) / std
    ).astype(
        np.float32
    )


# ============================================================
# METRICS
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_preds = []

    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = model(x)

            logits = output["logits"]

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item() * len(y)
            )

            total_samples += len(y)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_preds.extend(
                predictions.cpu().numpy()
            )

    avg_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    return avg_loss, accuracy, macro_f1


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print()
    print("=" * 70)
    print("EEG ENCODER TRAINING")
    print("=" * 70)

    print()
    print(f"Device          : {DEVICE}")
    print(f"Training subjects: {len(TRAIN_SUBJECTS)}")
    print(f"Validation subject: {VAL_SUBJECT}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Epochs          : {EPOCHS}")

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print()
    print("Loading training subjects...")

    train_eeg, train_labels, _, _ = (
        load_subjects(TRAIN_SUBJECTS)
    )

    print(
        f"Training EEG shape: "
        f"{train_eeg.shape}"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print()
    print("Loading validation subject...")

    val_eeg, val_labels, _, _ = (
        load_subjects([VAL_SUBJECT])
    )

    print(
        f"Validation EEG shape: "
        f"{val_eeg.shape}"
    )

    # --------------------------------------------------------
    # Verify expected sizes
    # --------------------------------------------------------

    assert train_eeg.shape == (
        23 * 27,
        28,
        8400
    )

    assert val_eeg.shape == (
        27,
        28,
        8400
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    print()
    print(
        "Calculating normalization "
        "from training subjects only..."
    )

    mean, std = calculate_normalization(
        train_eeg
    )

    train_eeg = normalize(
        train_eeg,
        mean,
        std
    )

    val_eeg = normalize(
        val_eeg,
        mean,
        std
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    train_x = torch.from_numpy(
        train_eeg
    )

    train_y = torch.from_numpy(
        train_labels
    )

    val_x = torch.from_numpy(
        val_eeg
    )

    val_y = torch.from_numpy(
        val_labels
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        train_x,
        train_y
    )

    val_dataset = TensorDataset(
        val_x,
        val_y
    )

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
    # Model
    # --------------------------------------------------------

    model = EEGEncoder(
        n_channels=28,
        n_classes=3,
        feature_dim=128,
        dropout=0.3
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_f1 = -1.0

    checkpoint_path = (
        CHECKPOINT_DIR /
        "eeg_encoder_best.pt"
    )

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
        total = 0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            output = model(x)

            logits = output["logits"]

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * len(y)
            )

            total += len(y)

        train_loss = (
            running_loss / total
        )

        val_loss, val_acc, val_f1 = (
            evaluate(
                model,
                val_loader
            )
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Save best model based on validation Macro-F1
        # ----------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "feature_dim":
                        128,

                    "n_channels":
                        28,

                    "n_classes":
                        3,

                    "dropout":
                        0.3,

                    "normalization_mean":
                        mean,

                    "normalization_std":
                        std,

                    "train_subjects":
                        TRAIN_SUBJECTS,

                    "validation_subject":
                        VAL_SUBJECT,

                    "best_val_macro_f1":
                        best_f1,

                    "seed":
                        SEED,
                },
                checkpoint_path
            )

            print(
                f"  -> Saved best checkpoint "
                f"(Macro-F1={best_f1:.4f})"
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EEG TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Macro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Checkpoint saved to:\n"
        f"{checkpoint_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()