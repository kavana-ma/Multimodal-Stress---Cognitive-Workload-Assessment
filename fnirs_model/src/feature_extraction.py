import h5py
import numpy as np

# ---------------------------------------
# Load preprocessed data
# ---------------------------------------
DATA_PATH = "../data/fnirs_all_subjects.mat"

with h5py.File(DATA_PATH, "r") as f:

    X = np.array(f["X_fnirs"])          # (2,300,36,648)
    y = np.array(f["y"]).squeeze()
    subject = np.array(f["subjectID"]).squeeze()

# Convert to (648,36,300,2)
X = np.transpose(X, (3,2,1,0))

# ---------------------------------------
# Feature extraction
# ---------------------------------------

features = []

for epoch in X:

    feat = []

    for signal in [0,1]:          # 0=HbO, 1=HbR

        data = epoch[:,:,signal]  # (36,300)

        for ch in range(36):

            s = data[ch]

            mean  = np.mean(s)
            std   = np.std(s)
            peak  = np.max(s)
            trough= np.min(s)

            slope = np.polyfit(np.arange(len(s)), s, 1)[0]

            auc   = np.trapz(s)

            feat.extend([mean,std,peak,trough,slope,auc])

    features.append(feat)

X_feat = np.array(features)

print("Feature matrix :", X_feat.shape)
print("Labels         :", y.shape)

# Save
np.savez(
    "../data/fnirs_features.npz",
    X=X_feat,
    y=y,
    subject=subject
)

print("Saved : fnirs_features.npz")