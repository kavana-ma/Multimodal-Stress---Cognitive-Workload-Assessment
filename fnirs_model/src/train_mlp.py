import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------------------------------
# Load feature dataset
# -------------------------------------------------
data = np.load("../data/fnirs_features.npz")

X = data["X"]              # (648,432)
y = data["y"]
subject = data["subject"]

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------------------------
# MLP Model
# -------------------------------------------------
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(432,256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,3)
        )

    def forward(self,x):
        return self.net(x)

# -------------------------------------------------
# LOSO Training
# -------------------------------------------------
results=[]

for test_subject in range(1,25):

    train_mask = subject != test_subject
    test_mask = subject == test_subject

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # Standardize using training only
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(
        FeatureDataset(X_train,y_train),
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        FeatureDataset(X_test,y_test),
        batch_size=32,
        shuffle=False
    )

    model = MLP().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    best_acc=0
    patience=8
    wait=0

    for epoch in range(50):

        model.train()

        for xb,yb in train_loader:

            xb=xb.to(device)
            yb=yb.to(device)

            optimizer.zero_grad()

            loss=criterion(model(xb),yb)

            loss.backward()

            optimizer.step()

        # Evaluation
        model.eval()

        correct=0
        total=0

        with torch.no_grad():

            for xb,yb in test_loader:

                xb=xb.to(device)
                yb=yb.to(device)

                pred=model(xb).argmax(1)

                correct+=(pred==yb).sum().item()
                total+=yb.size(0)

        acc=correct/total

        if acc>best_acc:
            best_acc=acc
            wait=0
        else:
            wait+=1

        if wait>=patience:
            break

    print(f"VP{test_subject:03d}: {best_acc*100:.2f}%")

    results.append(best_acc*100)

# -------------------------------------------------
# Final
# -------------------------------------------------
results=np.array(results)

print("\n"+"="*40)
print("FINAL LOSO RESULT")
print("="*40)
print("Mean Accuracy :", results.mean())
print("Std Accuracy  :", results.std())

pd.DataFrame({
    "Subject":[f"VP{i:03d}" for i in range(1,25)],
    "Accuracy":results
}).to_csv("../results/mlp_loso_results.csv",index=False)

print("Saved: mlp_loso_results.csv")