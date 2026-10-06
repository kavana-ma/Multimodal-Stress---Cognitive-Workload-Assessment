import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from attention_fusion_model import AttentionFusionModel


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
TEST_SIZE = 0.20
RANDOM_STATE = 42

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


# ============================================================
# Load fNIRS dataset
# ============================================================

fnirs_data = np.load(
    "results/fnirs_0vs3_dataset.npz"
)

fnirs_X = fnirs_data["X"]
fnirs_y = fnirs_data["y"]


# ============================================================
# Safety checks
# ============================================================

assert np.array_equal(eeg_y, fnirs_y), \
    "EEG and fNIRS labels do not match."

assert eeg_X.shape[0] == fnirs_X.shape[0], \
    "EEG and fNIRS sample counts do not match."


print("Total samples:", len(eeg_y))


# ============================================================
# Random 80/20 train-test split
# ============================================================

indices = np.arange(len(eeg_y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=eeg_y
)


print("\nRandom Split")
print("=" * 60)
print("Training samples:", len(train_idx))
print("Testing samples :", len(test_idx))

print("\nTraining class distribution:")
print(
    "0-back:",
    np.sum(eeg_y[train_idx] == 0)
)
print(
    "3-back:",
    np.sum(eeg_y[train_idx] == 1)
)

print("\nTesting class distribution:")
print(
    "0-back:",
    np.sum(eeg_y[test_idx] == 0)
)
print(
    "3-back:",
    np.sum(eeg_y[test_idx] == 1)
)


# ============================================================
# Split EEG
# ============================================================

eeg_train = eeg_X[train_idx]
eeg_test = eeg_X[test_idx]


# ============================================================
# Split fNIRS
# ============================================================

fnirs_train = fnirs_X[train_idx]
fnirs_test = fnirs_X[test_idx]


# ============================================================
# Labels
# ============================================================

y_train = eeg_y[train_idx]
y_test = eeg_y[test_idx]


# ============================================================
# Convert training data to tensors
# ============================================================

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


# ============================================================
# Convert test data to tensors
# ============================================================

eeg_test_tensor = torch.tensor(
    eeg_test,
    dtype=torch.float32
)

fnirs_test_tensor = torch.tensor(
    fnirs_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# Training DataLoader
# ============================================================

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


# ============================================================
# Create model
# ============================================================

model = AttentionFusionModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING ATTENTION FUSION")
print("=" * 60)

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

        optimizer.zero_grad()

        outputs, _, _, _ = model(
            eeg_batch,
            fnirs_batch
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

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


# ============================================================
# Evaluation
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATION")
print("=" * 60)

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


# ============================================================
# Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


# ============================================================
# Final result
# ============================================================

print("\n")
print("=" * 60)
print("RANDOM SPLIT ATTENTION FUSION RESULT")
print("=" * 60)

print(
    f"Train/Test Split: "
    f"{(1 - TEST_SIZE) * 100:.0f}/"
    f"{TEST_SIZE * 100:.0f}"
)

print(
    f"Random State: "
    f"{RANDOM_STATE}"
)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Mean EEG weight: "
    f"{np.mean(eeg_weights):.4f}"
)

print(
    f"Mean fNIRS weight: "
    f"{np.mean(fnirs_weights):.4f}"
)

print("=" * 60)