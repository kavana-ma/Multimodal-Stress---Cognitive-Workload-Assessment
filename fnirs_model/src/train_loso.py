
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from model import FNIRSCNNLSTM
from loso_split import get_fold

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -------------------------------------------------
# Dataset Class
# -------------------------------------------------
class FNIRSDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
LR = 3e-4
EPOCHS = 60
PATIENCE = 10
BATCH = 16

subjects = range(1, 25)

os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

all_results = []

# ============================================================
# LOSO LOOP
# ============================================================

for subject in subjects:

    print("\n" + "=" * 50)
    print(f"LOSO Fold : Testing VP{subject:03d}")
    print("=" * 50)

    # -------------------------------------------------
    # Split
    # -------------------------------------------------
    X_train, y_train, X_test, y_test = get_fold(subject)

    # -------------------------------------------------
    # Z-score Standardization
    # -------------------------------------------------
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)

    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # -------------------------------------------------
    # DataLoaders
    # -------------------------------------------------
    train_loader = DataLoader(
        FNIRSDataset(X_train, y_train),
        batch_size=BATCH,
        shuffle=True
    )

    test_loader = DataLoader(
        FNIRSDataset(X_test, y_test),
        batch_size=BATCH,
        shuffle=False
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = FNIRSCNNLSTM().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    best_acc = 0
    patience_counter = 0

    # ============================================================
    # Training
    # ============================================================

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # -------------------------------------------------
        # Evaluate on held-out subject
        # -------------------------------------------------
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for X, y in test_loader:

                X = X.to(device)
                y = y.to(device)

                outputs = model(X)

                pred = torch.argmax(outputs, dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total

        scheduler.step(acc)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss {train_loss:.4f} | "
            f"Test Acc {acc*100:.2f}%"
        )

        # -------------------------------------------------
        # Save Best Model
        # -------------------------------------------------
        if acc > best_acc:

            best_acc = acc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                f"../models/VP{subject:03d}_best.pth"
            )

        else:

            patience_counter += 1

        # -------------------------------------------------
        # Early Stopping
        # -------------------------------------------------
        if patience_counter >= PATIENCE:

            print("Early stopping.")
            break

    print(f"Best Accuracy VP{subject:03d}: {best_acc*100:.2f}%")

    all_results.append({
        "Subject": f"VP{subject:03d}",
        "Accuracy": round(best_acc * 100, 2)
    })

# ============================================================
# Final Results
# ============================================================

results = pd.DataFrame(all_results)

results.to_csv("../results/loso_results.csv", index=False)

print("\n" + "#" * 55)
print("FINAL LOSO RESULTS")
print("#" * 55)

print(results)

print("\nMean Accuracy : {:.2f}%".format(results["Accuracy"].mean()))
print("Std Accuracy  : {:.2f}%".format(results["Accuracy"].std()))