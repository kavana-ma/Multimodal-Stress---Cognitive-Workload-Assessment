import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from fnirs_cnn import FNIRSCNN


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "results/fnirs_0vs3_dataset.npz"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Load dataset
# ============================================================

data = np.load(DATA_PATH)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
subjects = data["subjects"]


unique_subjects = np.unique(subjects)

print("Dataset:", X.shape)
print("Subjects:", len(unique_subjects))


# ============================================================
# LOSO
# ============================================================

fold_accuracies = []
fold_balanced_acc = []
fold_f1 = []


for fold, test_subject in enumerate(
    unique_subjects,
    start=1
):

    print("\n" + "=" * 70)
    print(
        f"Fold {fold}/{len(unique_subjects)} "
        f"| Test subject: {test_subject}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Split by subject
    # --------------------------------------------------------

    train_mask = subjects != test_subject
    test_mask = subjects == test_subject

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print(
        "Train:",
        X_train.shape,
        "| Test:",
        X_test.shape
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FNIRSCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs, _ = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * batch_X.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.size(0)

        epoch_loss = (
            running_loss / total
        )

        epoch_acc = (
            correct / total
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Loss: {epoch_loss:.4f} "
            f"| Train Acc: {epoch_acc * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        test_tensor = X_test_tensor.to(
            DEVICE
        )

        outputs, embeddings = model(
            test_tensor
        )

        predictions = outputs.argmax(
            dim=1
        ).cpu().numpy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    fold_accuracies.append(
        accuracy
    )

    fold_balanced_acc.append(
        balanced_acc
    )

    fold_f1.append(
        f1
    )

    print("\nTest Results")
    print("-" * 40)

    print(
        f"Accuracy:          {accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: {balanced_acc * 100:.2f}%"
    )

    print(
        f"Weighted F1:       {f1 * 100:.2f}%"
    )


# ============================================================
# Final LOSO results
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL fNIRS LOSO RESULTS")
print("=" * 70)

mean_acc = np.mean(
    fold_accuracies
)

std_acc = np.std(
    fold_accuracies
)

mean_balanced = np.mean(
    fold_balanced_acc
)

mean_f1 = np.mean(
    fold_f1
)

print(
    f"Mean Accuracy:          "
    f"{mean_acc * 100:.2f}%"
)

print(
    f"Std Accuracy:           "
    f"{std_acc * 100:.2f}%"
)

print(
    f"Mean Balanced Accuracy: "
    f"{mean_balanced * 100:.2f}%"
)

print(
    f"Mean Weighted F1:       "
    f"{mean_f1 * 100:.2f}%"
)