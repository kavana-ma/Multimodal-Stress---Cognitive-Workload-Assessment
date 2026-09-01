import os
import numpy as np
from src.eeg.data.validation import validate_eeg_dataset


def make_synthetic(tmp_path):
    d = tmp_path
    d.mkdir(exist_ok=True)
    data_dir = str(d)
    N = 50
    C = 8
    T = 100
    X = np.random.randn(N, C, T).astype(np.float32)
    y = np.random.randint(0, 3, size=(N,))
    subject_id = np.array([f"S{int(i%5)}" for i in range(N)])
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
    return data_dir


def test_validate_synthetic(tmp_path):
    data_dir = make_synthetic(tmp_path)
    report = validate_eeg_dataset(data_dir)
    assert report["n_epochs"] == 50
    assert report["n_channels"] == 8
    assert report["n_samples"] == 100
