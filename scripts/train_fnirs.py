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

from models.fnirs_encoder import FNIRSEncoder


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


# Same development split used for EEG
VAL_SUBJECT = "VP026"

TRAIN_SUBJECTS = [
    s for s in SUBJECTS
    if s != VAL_SUBJECT
]


SEED = 42

BATCH_SIZE = 16
EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

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
# LOAD fNIRS FROM MATLAB V7.3
# ============================================================

def load_fnirs_subject(subject):

    path = (
        DATA_DIR
        / f"{subject}_nback_epochs.mat"
    )

    with h5py.File(path, "r") as f:

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
    # MATLAB v7.3 dimension order
    # --------------------------------------------------------
    #
    # MATLAB:
    #
    #   (27, 36, 420, 2)
    #
    # h5py:
    #
    #   (2, 420, 36, 27)
    #
    # Convert to:
    #
    #   (27, 36, 420, 2)
    # --------------------------------------------------------

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
        fnirs,
        labels,
        trial_id
    )


# ============================================================
# LOAD MULTIPLE SUBJECTS
# ============================================================

def load_subjects(subject_list):

    fnirs_list = []
    label_list = []
    trial_list = []

    for subject in subject_list:

        fnirs, labels, trial_id = (
            load_fnirs_subject(subject)
        )

        fnirs_list.append(fnirs)

        label_list.append(labels)

        trial_list.append(trial_id)

    fnirs = np.concatenate(
        fnirs_list,
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
        fnirs,
        labels,
        trials
    )


# ============================================================
# TRAINING-ONLY NORMALIZATION
# ============================================================

def calculate_normalization(fnirs):

    """
    Calculate channel/chromophore-wise statistics
    using training subjects only.

    Input:
        (N, 36, 420, 2)

    Statistics:
        one mean/std for each of the
        36 channels × 2 chromophores.
    """

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


def normalize(
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
# METRICS
# ============================================================

def evaluate(
    model,
    loader
):

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
        total_loss
        / total_samples
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

    return (
        avg_loss,
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
    print("fNIRS ENCODER TRAINING")
    print("=" * 70)

    print()
    print(
        f"Device           : {DEVICE}"
    )

    print(
        f"Training subjects: "
        f"{len(TRAIN_SUBJECTS)}"
    )

    print(
        f"Validation subject: "
        f"{VAL_SUBJECT}"
    )

    print(
        f"Batch size       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs           : "
        f"{EPOCHS}"
    )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print()
    print(
        "Loading training subjects..."
    )

    train_fnirs, train_labels, _ = (
        load_subjects(
            TRAIN_SUBJECTS
        )
    )

    print(
        f"Training fNIRS shape: "
        f"{train_fnirs.shape}"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print()
    print(
        "Loading validation subject..."
    )

    val_fnirs, val_labels, _ = (
        load_subjects(
            [VAL_SUBJECT]
        )
    )

    print(
        f"Validation fNIRS shape: "
        f"{val_fnirs.shape}"
    )

    # --------------------------------------------------------
    # Verify sizes
    # --------------------------------------------------------

    assert train_fnirs.shape == (
        23 * 27,
        36,
        420,
        2
    )

    assert val_fnirs.shape == (
        27,
        36,
        420,
        2
    )

    # --------------------------------------------------------
    # Verify finite input
    # --------------------------------------------------------

    assert np.isfinite(
        train_fnirs
    ).all()

    assert np.isfinite(
        val_fnirs
    ).all()

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    print()
    print(
        "Calculating normalization "
        "from training subjects only..."
    )

    mean, std = (
        calculate_normalization(
            train_fnirs
        )
    )

    train_fnirs = normalize(
        train_fnirs,
        mean,
        std
    )

    val_fnirs = normalize(
        val_fnirs,
        mean,
        std
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    train_x = torch.from_numpy(
        train_fnirs
    )

    train_y = torch.from_numpy(
        train_labels
    )

    val_x = torch.from_numpy(
        val_fnirs
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

    model = FNIRSEncoder(
        n_channels=36,
        n_chromophores=2,
        n_classes=3,
        feature_dim=128,
        dropout=0.3
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

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
        CHECKPOINT_DIR
        / "fnirs_encoder_best.pt"
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
                loss.item()
                * len(y)
            )

            total += len(y)

        train_loss = (
            running_loss
            / total
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
        # Save best checkpoint
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
                        36,

                    "n_chromophores":
                        2,

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
    print("fNIRS TRAINING COMPLETE")
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