import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from attention_fusion_model import AttentionFusionModel


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Load EEG dataset
# ============================================================

eeg_data = np.load(
    "results/eeg_0vs3_dataset.npz"
)

eeg_X = eeg_data["X"]
eeg_y = eeg_data["y"]
eeg_subjects = eeg_data["subjects"]


# ============================================================
# Load fNIRS dataset
# ============================================================

fnirs_data = np.load(
    "results/fnirs_0vs3_dataset.npz"
)

fnirs_X = fnirs_data["X"]
fnirs_y = fnirs_data["y"]
fnirs_subjects = fnirs_data["subjects"]


# ============================================================
# Safety checks
# ============================================================

assert np.array_equal(eeg_y, fnirs_y), \
    "EEG and fNIRS labels do not match."

assert np.array_equal(eeg_subjects, fnirs_subjects), \
    "EEG and fNIRS subjects do not match."

assert eeg_X.shape[0] == fnirs_X.shape[0], \
    "EEG and fNIRS sample counts do not match."


# ============================================================
# Subjects
# ============================================================

subjects = np.unique(eeg_subjects)

print("Number of subjects:", len(subjects))
print("Number of samples:", len(eeg_y))


# ============================================================
# Store LOSO results
# ============================================================

accuracies = []
balanced_accuracies = []
f1_scores = []


# ============================================================
# LOSO
# ============================================================

for fold, test_subject in enumerate(subjects, start=1):

    print("\n" + "=" * 60)
    print(
        f"FOLD {fold}/{len(subjects)} "
        f"| Test Subject: {test_subject}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Train / test masks
    # --------------------------------------------------------

    train_mask = eeg_subjects != test_subject
    test_mask = eeg_subjects == test_subject

    # --------------------------------------------------------
    # Split EEG
    # --------------------------------------------------------

    eeg_train = eeg_X[train_mask]
    eeg_test = eeg_X[test_mask]

    # --------------------------------------------------------
    # Split fNIRS
    # --------------------------------------------------------

    fnirs_train = fnirs_X[train_mask]
    fnirs_test = fnirs_X[test_mask]

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_train = eeg_y[train_mask]
    y_test = eeg_y[test_mask]

    print("Train samples:", len(y_train))
    print("Test samples :", len(y_test))

    # --------------------------------------------------------
    # Convert training data to tensors
    # --------------------------------------------------------

    eeg_train_tensor = torch.tensor(
        eeg_train,
        dtype=torch.float32
    )

    fnirs_train_tensor = torch.tensor(
        fnirs_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Convert test data to tensors
    # --------------------------------------------------------

    eeg_test_tensor = torch.tensor(
        eeg_test,
        dtype=torch.float32
    )

    fnirs_test_tensor = torch.tensor(
        fnirs_test,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Training DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        eeg_train_tensor,
        fnirs_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------------
    # Create fresh model for this LOSO fold
    # --------------------------------------------------------

    model = AttentionFusionModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for (
            eeg_batch,
            fnirs_batch,
            labels
        ) in train_loader:

            eeg_batch = eeg_batch.to(DEVICE)
            fnirs_batch = fnirs_batch.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _, _, _ = model(
                eeg_batch,
                fnirs_batch
            )

            # Classification loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss /
            len(train_loader)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Loss: {avg_loss:.4f}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        eeg_test_tensor = eeg_test_tensor.to(DEVICE)
        fnirs_test_tensor = fnirs_test_tensor.to(DEVICE)

        (
            outputs,
            _,
            eeg_weights,
            fnirs_weights
        ) = model(
            eeg_test_tensor,
            fnirs_test_tensor
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

        # Store learned modality weights for this fold
        eeg_weights = (
            eeg_weights
            .squeeze(1)
            .cpu()
            .numpy()
        )

        fnirs_weights = (
            fnirs_weights
            .squeeze(1)
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    accuracies.append(accuracy)
    balanced_accuracies.append(
        balanced_accuracy
    )
    f1_scores.append(f1)

    # --------------------------------------------------------
    # Fold results
    # --------------------------------------------------------

    print("\nFold Results:")

    print(
        f"Accuracy:          "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Weighted F1:       "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Mean EEG weight:   "
        f"{np.mean(eeg_weights):.4f}"
    )

    print(
        f"Mean fNIRS weight: "
        f"{np.mean(fnirs_weights):.4f}"
    )


# ============================================================
# Final results
# ============================================================

print("\n")
print("=" * 60)
print("FINAL ATTENTION FUSION LOSO RESULTS")
print("=" * 60)

print(
    f"Mean Accuracy: "
    f"{np.mean(accuracies) * 100:.2f}%"
)

print(
    f"Std Accuracy: "
    f"{np.std(accuracies) * 100:.2f}%"
)

print(
    f"Mean Balanced Accuracy: "
    f"{np.mean(balanced_accuracies) * 100:.2f}%"
)

print(
    f"Mean Weighted F1: "
    f"{np.mean(f1_scores) * 100:.2f}%"
)

print("=" * 60)