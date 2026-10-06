import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from eeg_cnn import EEGCNN


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "results/eeg_0vs3_dataset.npz"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(DATA_FILE)

X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
subjects = data["subjects"]


print("=" * 70)
print("EEG CNN - LOSO TRAINING")
print("=" * 70)

print("Dataset shape:", X.shape)
print("Number of subjects:", len(np.unique(subjects)))
print("Device:", DEVICE)
print()


# ============================================================
# LOSO
# ============================================================

unique_subjects = np.unique(subjects)

fold_results = []


for fold, test_subject in enumerate(unique_subjects, start=1):

    print("=" * 70)
    print(f"FOLD {fold}/{len(unique_subjects)}")
    print(f"Test subject: {test_subject}")
    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_mask = subjects != test_subject
    test_mask = subjects == test_subject

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]


    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))


    # --------------------------------------------------------
    # CONVERT TO TORCH
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


    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = EEGCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    for epoch in range(EPOCHS):

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            output, _ = model(batch_X)

            loss = criterion(
                output,
                batch_y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * batch_X.size(0)
            )

            predictions = torch.argmax(
                output,
                dim=1
            )

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.size(0)


        epoch_loss = running_loss / total
        epoch_accuracy = correct / total


        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_accuracy:.4f}"
        )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    predictions = []
    true_labels = []


    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(DEVICE)

            output, _ = model(batch_X)

            pred = torch.argmax(
                output,
                dim=1
            )

            predictions.extend(
                pred.cpu().numpy()
            )

            true_labels.extend(
                batch_y.numpy()
            )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predictions
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average="weighted"
    )


    print()
    print("Test Accuracy:", accuracy)
    print("Balanced Accuracy:", balanced_accuracy)
    print("F1 Score:", f1)


    fold_results.append({
        "subject": test_subject,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1
    })


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL LOSO RESULTS")
print("=" * 70)

accuracies = [
    result["accuracy"]
    for result in fold_results
]

balanced_accuracies = [
    result["balanced_accuracy"]
    for result in fold_results
]

f1_scores = [
    result["f1"]
    for result in fold_results
]


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
    f"Mean F1 Score: "
    f"{np.mean(f1_scores) * 100:.2f}%"
)