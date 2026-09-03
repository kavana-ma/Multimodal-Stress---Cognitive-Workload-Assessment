import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load features
# -------------------------------------------------
data = np.load("../data/fnirs_features.npz")

X = data["X"]
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
# MLP
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
# LOSO Evaluation
# -------------------------------------------------
all_true = []
all_pred = []

for test_subject in range(1,25):

    train_mask = subject != test_subject
    test_mask = subject == test_subject

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

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

    best_acc = 0
    best_state = None
    wait = 0

    for epoch in range(50):

        model.train()

        for xb,yb in train_loader:

            xb=xb.to(device)
            yb=yb.to(device)

            optimizer.zero_grad()

            loss = criterion(model(xb),yb)

            loss.backward()

            optimizer.step()

        # validation on held-out subject
        model.eval()

        pred_epoch=[]

        with torch.no_grad():

            for xb,yb in test_loader:

                xb=xb.to(device)

                pred=model(xb).argmax(1).cpu().numpy()

                pred_epoch.extend(pred)

        acc=accuracy_score(y_test,pred_epoch)

        if acc>best_acc:

            best_acc=acc
            best_state=model.state_dict()
            wait=0

        else:
            wait+=1

        if wait>=8:
            break

    # reload best model
    model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():

        for xb,yb in test_loader:

            xb=xb.to(device)

            pred=model(xb).argmax(1).cpu().numpy()

            all_pred.extend(pred)
            all_true.extend(yb.numpy())

# -------------------------------------------------
# Results
# -------------------------------------------------
acc = accuracy_score(all_true,all_pred)

print("\nOverall Accuracy :",round(acc*100,2),"%")

print("\nClassification Report\n")
print(classification_report(
    all_true,
    all_pred,
    target_names=["Low","Medium","High"]
))

cm = confusion_matrix(all_true,all_pred)

plt.figure(figsize=(5,4))
plt.imshow(cm,cmap="Blues")

plt.xticks([0,1,2],["Low","Medium","High"])
plt.yticks([0,1,2],["Low","Medium","High"])

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

for i in range(3):
    for j in range(3):
        plt.text(j,i,str(cm[i,j]),
                 ha="center",va="center",fontsize=12)

plt.tight_layout()

os.makedirs("../results",exist_ok=True)

plt.savefig("../results/confusion_matrix.png",dpi=300)

print("\nSaved: results/confusion_matrix.png")