import os
import numpy as np
import tempfile
import shutil

from src.eeg.data.validation import validate_eeg_dataset
from src.eeg.data.splitting import subject_group_kfold_indices
from src.eeg.preprocessing.normalization import LeakSafeScaler
from src.eeg.data.dataset import EEGDataset
from src.eeg.models.eeg_cnn import SimpleEEGCNN


def make_synthetic(data_dir):
    N = 60
    C = 8
    T = 120
    X = np.random.randn(N, C, T).astype(np.float32)
    y = np.random.randint(0, 3, size=(N,))
    subject_id = np.array([f"S{int(i%6)}" for i in range(N)])
    trial_id = np.array([f"trial_{i}" for i in range(N)])
    np.save(os.path.join(data_dir, "X.npy"), X)
    np.save(os.path.join(data_dir, "y.npy"), y)
    np.save(os.path.join(data_dir, "subject_id.npy"), subject_id)
    np.save(os.path.join(data_dir, "trial_id.npy"), trial_id)
    import json
    json.dump([f"ch{i}" for i in range(C)], open(os.path.join(data_dir, "channel_names.json"), "w"))
    json.dump({"0":0,"1":1,"2":2}, open(os.path.join(data_dir, "label_map.json"), "w"))
    import pandas as pd
    pd.DataFrame({"epoch_id": list(range(N)), "label": y}).to_csv(os.path.join(data_dir, "metadata.csv"), index=False)


def run():
    tmp = tempfile.mkdtemp()
    try:
        make_synthetic(tmp)
        report = validate_eeg_dataset(tmp)
        print("VALIDATION_OK", report["n_epochs"], report["n_channels"], report["n_samples"])

        X = np.load(os.path.join(tmp, "X.npy"))
        subject_id = np.load(os.path.join(tmp, "subject_id.npy"))
        folds = subject_group_kfold_indices(subject_id, n_splits=3)
        print("SPLITS_OK", len(folds))

        train_idx, test_idx = folds[0]
        scaler = LeakSafeScaler()
        scaler.fit(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])
        print("SCALER_OK", X_test_scaled.shape)

        ds = EEGDataset(X_test_scaled, np.load(os.path.join(tmp, "y.npy"))[test_idx], subject_id[test_idx], np.load(os.path.join(tmp, "trial_id.npy"))[test_idx])
        item = ds[0]
        print("DATASET_OK", item["x"].shape)

        import torch
        model = SimpleEEGCNN(n_channels=report["n_channels"], n_samples=report["n_samples"], embedding_dim=16, num_classes=3)
        x_batch = torch.stack([item["x"] for _ in range(2)])
        emb, logits = model(x_batch)
        print("MODEL_OK", emb.shape, logits.shape)

        print("SANITY_OK")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    run()
