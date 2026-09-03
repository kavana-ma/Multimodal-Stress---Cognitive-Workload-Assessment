import numpy as np
import h5py

# Load complete dataset
with h5py.File("../data/fnirs_all_subjects.mat","r") as f:
    X = np.array(f["X_fnirs"])
    y = np.array(f["y"]).squeeze()
    subject = np.array(f["subjectID"]).squeeze()

X = np.transpose(X,(3,2,1,0))

# reshape into (N,300,72)
hbo = np.transpose(X[:,:,:,0],(0,2,1))
hbr = np.transpose(X[:,:,:,1],(0,2,1))
X = np.concatenate([hbo,hbr],axis=2)

def get_fold(test_subject):

    train_mask = subject != test_subject
    test_mask = subject == test_subject

    return (
        X[train_mask],
        y[train_mask],
        X[test_mask],
        y[test_mask]
    )