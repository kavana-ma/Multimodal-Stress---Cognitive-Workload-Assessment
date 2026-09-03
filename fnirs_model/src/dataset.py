"""
03_dataset.py
Prepare fNIRS dataset for CNN-LSTM
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------
# Load subject-wise split
# ---------------------------------------

data = np.load("../data/fnirs_subject_split.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

# ---------------------------------------
# Reshape: (N,36,300,2) -> (N,300,72)
# ---------------------------------------

def reshape_tensor(X):

    hbo = X[:, :, :, 0]              # (N,36,300)
    hbr = X[:, :, :, 1]

    hbo = np.transpose(hbo, (0,2,1)) # (N,300,36)
    hbr = np.transpose(hbr, (0,2,1))

    X_new = np.concatenate([hbo, hbr], axis=2)

    return X_new

X_train = reshape_tensor(X_train)
X_val = reshape_tensor(X_val)
X_test = reshape_tensor(X_test)

print("After reshape")
print("----------------")
print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Test  :", X_test.shape)

# ---------------------------------------
# PyTorch Dataset
# ---------------------------------------

class FNIRSDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FNIRSDataset(X_train, y_train)
val_dataset = FNIRSDataset(X_val, y_val)
test_dataset = FNIRSDataset(X_test, y_test)

train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True)

val_loader = DataLoader(val_dataset,
                        batch_size=32,
                        shuffle=False)

test_loader = DataLoader(test_dataset,
                         batch_size=32,
                         shuffle=False)

# ---------------------------------------
# Verify one batch
# ---------------------------------------

X_batch, y_batch = next(iter(train_loader))

print("\nBatch verification")
print("------------------")
print("Input :", X_batch.shape)
print("Labels:", y_batch.shape)
print(X_train.dtype)
print(X_train.min(), X_train.max())
print(X_train.mean(), X_train.std())