# scripts/run_loso.py

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.eeg_encoder import EEGEncoder
from models.fnirs_encoder import FNIRSEncoder
from models.fusion_model import FusionModel


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ALL_SUBJECTS = [
    "VP001",
    "VP002",
    "VP003",
    "VP004",
    "VP005",
    "VP006",
    "VP007",
    "VP008",
    "VP009",
    "VP010",
    "VP011",
    "VP014",
    "VP015",
    "VP016",
    "VP017",
    "VP018",
    "VP019",
    "VP020",
    "VP021",
    "VP022",
    "VP023",
    "VP024",
    "VP025",
    "VP026",
]

N_CLASSES = 3

EEG_CHANNELS = 28
EEG_SAMPLES = 8400

FNIRS_CHANNELS = 36
FNIRS_SAMPLES = 420
FNIRS_CHROMOPHORES = 2

EEG_FEATURE_DIM = 128
FNIRS_FEATURE_DIM = 128
FUSION_DIM = 128

DROPOUT = 0.3

# Final training settings
EEG_EPOCHS = 30
FNIRS_EPOCHS = 30
FUSION_EPOCHS = 40

EEG_BATCH_SIZE = 8
FNIRS_BATCH_SIZE = 16
FUSION_BATCH_SIZE = 16

LR = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "epochs"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "loso"
)

CHECKPOINT_DIR = os.path.join(
    RESULTS_DIR,
    "checkpoints"
)

PREDICTION_DIR = os.path.join(
    RESULTS_DIR,
    "predictions"
)


# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# LOAD MATLAB EPOCH FILE
# ---------------------------------------------------------

def load_subject(subject_id):

    path = os.path.join(
        DATA_DIR,
        f"{subject_id}_nback_epochs.mat"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:

        eeg = np.array(
            f["eeg_epochs"]
        )

        fnirs = np.array(
            f["fnirs_epochs"]
        )

        labels = np.array(
            f["labels"]
        ).squeeze()

        trial_id = np.array(
            f["trial_id"]
        ).squeeze()

    # -----------------------------------------------------
    # MATLAB -> PYTHON AXES
    #
    # EEG MATLAB:
    #   channels × samples × epochs
    #
    # Python:
    #   epochs × channels × samples
    # -----------------------------------------------------

    if eeg.ndim == 3:

        # Expected HDF5 shape:
        # (8400, 28, 27)

        if eeg.shape[0] == EEG_SAMPLES:
            eeg = np.transpose(
                eeg,
                (2, 1, 0)
            )

        elif eeg.shape[1] == EEG_CHANNELS:
            # Already probably epochs x channels x samples
            pass

        else:
            raise ValueError(
                f"Unexpected EEG shape for {subject_id}: "
                f"{eeg.shape}"
            )

    # -----------------------------------------------------
    # fNIRS MATLAB:
    #   chromophores × samples × channels × epochs
    #
    # Python:
    #   epochs × channels × samples × chromophores
    # -----------------------------------------------------

    if fnirs.ndim == 4:

        # Expected:
        # (2, 420, 36, 27)

        if (
            fnirs.shape[0] == FNIRS_CHROMOPHORES
            and fnirs.shape[1] == FNIRS_SAMPLES
            and fnirs.shape[2] == FNIRS_CHANNELS
        ):

            fnirs = np.transpose(
                fnirs,
                (3, 2, 1, 0)
            )

        elif (
            fnirs.shape[1] == FNIRS_CHANNELS
            and fnirs.shape[2] == FNIRS_SAMPLES
        ):
            # Already correct
            pass

        else:
            raise ValueError(
                f"Unexpected fNIRS shape for {subject_id}: "
                f"{fnirs.shape}"
            )

    labels = labels.astype(np.int64)
    trial_id = trial_id.astype(np.int64)

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if eeg.shape != (
        27,
        EEG_CHANNELS,
        EEG_SAMPLES
    ):
        raise ValueError(
            f"EEG shape incorrect for {subject_id}: "
            f"{eeg.shape}"
        )

    if fnirs.shape != (
        27,
        FNIRS_CHANNELS,
        FNIRS_SAMPLES,
        FNIRS_CHROMOPHORES
    ):
        raise ValueError(
            f"fNIRS shape incorrect for {subject_id}: "
            f"{fnirs.shape}"
        )

    if labels.shape[0] != 27:
        raise ValueError(
            f"Expected 27 labels for {subject_id}, "
            f"got {labels.shape}"
        )

    if np.isnan(eeg).any() or np.isinf(eeg).any():
        raise ValueError(
            f"NaN/Inf detected in EEG: {subject_id}"
        )

    if np.isnan(fnirs).any() or np.isinf(fnirs).any():
        raise ValueError(
            f"NaN/Inf detected in fNIRS: {subject_id}"
        )

    return (
        eeg.astype(np.float32),
        fnirs.astype(np.float32),
        labels,
        trial_id
    )


# ---------------------------------------------------------
# LOAD MULTIPLE SUBJECTS
# ---------------------------------------------------------

def load_subjects(subjects):

    eeg_list = []
    fnirs_list = []
    labels_list = []
    trial_list = []
    subject_list = []

    for subject in subjects:

        eeg, fnirs, labels, trial_id = load_subject(
            subject
        )

        eeg_list.append(eeg)
        fnirs_list.append(fnirs)
        labels_list.append(labels)
        trial_list.append(trial_id)

        subject_list.extend(
            [subject] * len(labels)
        )

    eeg = np.concatenate(
        eeg_list,
        axis=0
    )

    fnirs = np.concatenate(
        fnirs_list,
        axis=0
    )

    labels = np.concatenate(
        labels_list,
        axis=0
    )

    trial_id = np.concatenate(
        trial_list,
        axis=0
    )

    subject_list = np.array(
        subject_list
    )

    return (
        eeg,
        fnirs,
        labels,
        trial_id,
        subject_list
    )


# ---------------------------------------------------------
# TRAINING-ONLY NORMALIZATION
# ---------------------------------------------------------

def calculate_normalization(
    eeg_train,
    fnirs_train
):

    # EEG:
    # channel-wise mean/std
    #
    # Shape:
    # (28,)

    eeg_mean = eeg_train.mean(
        axis=(0, 2),
        keepdims=False
    )

    eeg_std = eeg_train.std(
        axis=(0, 2),
        keepdims=False
    )

    eeg_std[eeg_std < 1e-8] = 1.0

    # fNIRS:
    # channel + chromophore-wise
    #
    # Shape:
    # (36, 2)

    fnirs_mean = fnirs_train.mean(
        axis=(0, 2)
    )

    fnirs_std = fnirs_train.std(
        axis=(0, 2)
    )

    fnirs_std[fnirs_std < 1e-8] = 1.0

    return (
        eeg_mean.astype(np.float32),
        eeg_std.astype(np.float32),
        fnirs_mean.astype(np.float32),
        fnirs_std.astype(np.float32)
    )


def normalize_eeg(
    eeg,
    mean,
    std
):

    # eeg:
    # (N, 28, 8400)

    mean = mean.reshape(
        1,
        EEG_CHANNELS,
        1
    )

    std = std.reshape(
        1,
        EEG_CHANNELS,
        1
    )

    return (
        (eeg - mean) / std
    ).astype(np.float32)


def normalize_fnirs(
    fnirs,
    mean,
    std
):

    # fnirs:
    # (N, 36, 420, 2)

    mean = mean.reshape(
        1,
        FNIRS_CHANNELS,
        1,
        FNIRS_CHROMOPHORES
    )

    std = std.reshape(
        1,
        FNIRS_CHANNELS,
        1,
        FNIRS_CHROMOPHORES
    )

    return (
        (fnirs - mean) / std
    ).astype(np.float32)


# ---------------------------------------------------------
# MODEL OUTPUT HELPERS
# ---------------------------------------------------------

def get_logits(output):

    if isinstance(output, dict):

        if "logits" in output:
            return output["logits"]

        raise ValueError(
            "Model output dictionary does not contain logits."
        )

    if isinstance(output, tuple):
        return output[-1]

    return output


def get_features(output):

    if isinstance(output, dict):

        if "features" in output:
            return output["features"]

        raise ValueError(
            "Model output dictionary does not contain features."
        )

    if isinstance(output, tuple):
        return output[0]

    return output


# ---------------------------------------------------------
# TRAIN EEG ENCODER
# ---------------------------------------------------------

def train_eeg_encoder(
    x_train,
    y_train,
    x_val,
    y_val
):

    model = EEGEncoder(
        n_channels=EEG_CHANNELS,
        n_classes=N_CLASSES,
        feature_dim=EEG_FEATURE_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=EEG_BATCH_SIZE,
        shuffle=True
    )

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EEG_EPOCHS + 1):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            output = model(xb)

            logits = get_logits(output)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():

            xb = torch.from_numpy(
                x_val
            ).to(DEVICE)

            output = model(xb)

            logits = get_logits(output)

            predictions = (
                torch.argmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

        val_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(
        best_state
    )

    return model, best_f1


# ---------------------------------------------------------
# TRAIN fNIRS ENCODER
# ---------------------------------------------------------

def train_fnirs_encoder(
    x_train,
    y_train,
    x_val,
    y_val
):

    model = FNIRSEncoder(
        n_channels=FNIRS_CHANNELS,
        n_chromophores=FNIRS_CHROMOPHORES,
        n_classes=N_CLASSES,
        feature_dim=FNIRS_FEATURE_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=FNIRS_BATCH_SIZE,
        shuffle=True
    )

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, FNIRS_EPOCHS + 1):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            output = model(xb)

            logits = get_logits(output)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():

            xb = torch.from_numpy(
                x_val
            ).to(DEVICE)

            output = model(xb)

            logits = get_logits(output)

            predictions = (
                torch.argmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

        val_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(
        best_state
    )

    return model, best_f1


# ---------------------------------------------------------
# TRAIN FUSION MODEL
# ---------------------------------------------------------

def train_fusion_model(
    eeg_encoder,
    fnirs_encoder,
    eeg_train,
    fnirs_train,
    y_train,
    eeg_val,
    fnirs_val,
    y_val
):

    model = FusionModel(
        eeg_encoder=eeg_encoder,
        fnirs_encoder=fnirs_encoder,
        eeg_dim=EEG_FEATURE_DIM,
        fnirs_dim=FNIRS_FEATURE_DIM,
        fusion_dim=FUSION_DIM,
        n_classes=N_CLASSES,
        dropout=DROPOUT,
        freeze_encoders=True
    ).to(DEVICE)

    # Only projection + classifier parameters
    # should be trainable.
    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(eeg_train),
        torch.from_numpy(fnirs_train),
        torch.from_numpy(y_train)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=FUSION_BATCH_SIZE,
        shuffle=True
    )

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, FUSION_EPOCHS + 1):

        model.train()

        for eeg_x, fnirs_x, yb in train_loader:

            eeg_x = eeg_x.to(DEVICE)
            fnirs_x = fnirs_x.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            output = model(
                eeg_x,
                fnirs_x
            )

            logits = get_logits(output)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():

            eeg_x = torch.from_numpy(
                eeg_val
            ).to(DEVICE)

            fnirs_x = torch.from_numpy(
                fnirs_val
            ).to(DEVICE)

            output = model(
                eeg_x,
                fnirs_x
            )

            logits = get_logits(output)

            predictions = (
                torch.argmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

        val_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(
        best_state
    )

    return model, best_f1


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(
    model,
    x,
    y
):

    model.eval()

    all_probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(x),
            32
        ):

            end = min(
                start + 32,
                len(x)
            )

            xb = torch.from_numpy(
                x[start:end]
            ).to(DEVICE)

            output = model(xb)

            logits = get_logits(output)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities
    )

    return (
        metrics,
        predictions,
        probabilities
    )


# ---------------------------------------------------------
# EEG EVALUATION
# ---------------------------------------------------------

def evaluate_eeg(
    model,
    eeg_x,
    y
):

    model.eval()

    probabilities_list = []

    with torch.no_grad():

        for start in range(
            0,
            len(eeg_x),
            32
        ):

            end = min(
                start + 32,
                len(eeg_x)
            )

            xb = torch.from_numpy(
                eeg_x[start:end]
            ).to(DEVICE)

            output = model(xb)

            logits = get_logits(output)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            probabilities_list.append(
                probabilities
                .cpu()
                .numpy()
            )

    probabilities = np.concatenate(
        probabilities_list,
        axis=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities
    )

    return (
        metrics,
        predictions,
        probabilities
    )


# ---------------------------------------------------------
# fNIRS EVALUATION
# ---------------------------------------------------------

def evaluate_fnirs(
    model,
    fnirs_x,
    y
):

    model.eval()

    probabilities_list = []

    with torch.no_grad():

        for start in range(
            0,
            len(fnirs_x),
            32
        ):

            end = min(
                start + 32,
                len(fnirs_x)
            )

            xb = torch.from_numpy(
                fnirs_x[start:end]
            ).to(DEVICE)

            output = model(xb)

            logits = get_logits(output)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            probabilities_list.append(
                probabilities
                .cpu()
                .numpy()
            )

    probabilities = np.concatenate(
        probabilities_list,
        axis=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities
    )

    return (
        metrics,
        predictions,
        probabilities
    )


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        roc_auc = np.nan

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


# ---------------------------------------------------------
# RUN ONE LOSO FOLD
# ---------------------------------------------------------

def run_fold(
    test_subject,
    fold_index
):

    test_index = ALL_SUBJECTS.index(
        test_subject
    )

    # Next subject cyclically = validation
    val_subject = ALL_SUBJECTS[
        (test_index + 1) % len(ALL_SUBJECTS)
    ]

    train_subjects = [
        s for s in ALL_SUBJECTS
        if s not in [
            test_subject,
            val_subject
        ]
    ]

    print()
    print("=" * 70)
    print(
        f"FOLD {fold_index + 1}/"
        f"{len(ALL_SUBJECTS)}"
    )
    print("=" * 70)

    print(
        f"Test subject:       {test_subject}"
    )

    print(
        f"Validation subject: {val_subject}"
    )

    print(
        f"Training subjects:   {len(train_subjects)}"
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    (
        eeg_train,
        fnirs_train,
        y_train,
        _,
        _
    ) = load_subjects(
        train_subjects
    )

    (
        eeg_val,
        fnirs_val,
        y_val,
        _,
        _
    ) = load_subjects(
        [val_subject]
    )

    (
        eeg_test,
        fnirs_test,
        y_test,
        trial_id_test,
        subject_test
    ) = load_subjects(
        [test_subject]
    )

    print(
        f"Train EEG:   {eeg_train.shape}"
    )

    print(
        f"Train fNIRS: {fnirs_train.shape}"
    )

    print(
        f"Val EEG:     {eeg_val.shape}"
    )

    print(
        f"Val fNIRS:   {fnirs_val.shape}"
    )

    print(
        f"Test EEG:    {eeg_test.shape}"
    )

    print(
        f"Test fNIRS:  {fnirs_test.shape}"
    )

    # -----------------------------------------------------
    # TRAINING-ONLY NORMALIZATION
    # -----------------------------------------------------

    print()
    print(
        "Calculating training-only normalization..."
    )

    (
        eeg_mean,
        eeg_std,
        fnirs_mean,
        fnirs_std
    ) = calculate_normalization(
        eeg_train,
        fnirs_train
    )

    eeg_train = normalize_eeg(
        eeg_train,
        eeg_mean,
        eeg_std
    )

    eeg_val = normalize_eeg(
        eeg_val,
        eeg_mean,
        eeg_std
    )

    eeg_test = normalize_eeg(
        eeg_test,
        eeg_mean,
        eeg_std
    )

    fnirs_train = normalize_fnirs(
        fnirs_train,
        fnirs_mean,
        fnirs_std
    )

    fnirs_val = normalize_fnirs(
        fnirs_val,
        fnirs_mean,
        fnirs_std
    )

    fnirs_test = normalize_fnirs(
        fnirs_test,
        fnirs_mean,
        fnirs_std
    )

    # -----------------------------------------------------
    # EEG
    # -----------------------------------------------------

    print()
    print(
        "TRAINING EEG ENCODER"
    )

    eeg_encoder, eeg_val_f1 = train_eeg_encoder(
        eeg_train,
        y_train,
        eeg_val,
        y_val
    )

    print(
        f"Best EEG Val Macro-F1: "
        f"{eeg_val_f1:.4f}"
    )

    # -----------------------------------------------------
    # fNIRS
    # -----------------------------------------------------

    print()
    print(
        "TRAINING fNIRS ENCODER"
    )

    fnirs_encoder, fnirs_val_f1 = train_fnirs_encoder(
        fnirs_train,
        y_train,
        fnirs_val,
        y_val
    )

    print(
        f"Best fNIRS Val Macro-F1: "
        f"{fnirs_val_f1:.4f}"
    )

    # -----------------------------------------------------
    # FUSION
    # -----------------------------------------------------

    print()
    print(
        "TRAINING FUSION MODEL"
    )

    fusion_model, fusion_val_f1 = train_fusion_model(
        eeg_encoder,
        fnirs_encoder,
        eeg_train,
        fnirs_train,
        y_train,
        eeg_val,
        fnirs_val,
        y_val
    )

    print(
        f"Best Fusion Val Macro-F1: "
        f"{fusion_val_f1:.4f}"
    )

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    print()
    print(
        f"FINAL TEST: {test_subject}"
    )

    # EEG
    eeg_metrics, eeg_pred, eeg_prob = evaluate_eeg(
        eeg_encoder,
        eeg_test,
        y_test
    )

    # fNIRS
    fnirs_metrics, fnirs_pred, fnirs_prob = evaluate_fnirs(
        fnirs_encoder,
        fnirs_test,
        y_test
    )

    # Fusion
    fusion_model.eval()

    fusion_probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(eeg_test),
            16
        ):

            end = min(
                start + 16,
                len(eeg_test)
            )

            eeg_x = torch.from_numpy(
                eeg_test[start:end]
            ).to(DEVICE)

            fnirs_x = torch.from_numpy(
                fnirs_test[start:end]
            ).to(DEVICE)

            output = fusion_model(
                eeg_x,
                fnirs_x
            )

            logits = get_logits(output)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            fusion_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )

    fusion_prob = np.concatenate(
        fusion_probabilities,
        axis=0
    )

    fusion_pred = np.argmax(
        fusion_prob,
        axis=1
    )

    fusion_metrics = calculate_metrics(
        y_test,
        fusion_pred,
        fusion_prob
    )

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print()
    print("EEG-only")
    print(
        f"Accuracy:          "
        f"{eeg_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro Precision:   "
        f"{eeg_metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall:      "
        f"{eeg_metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1:          "
        f"{eeg_metrics['macro_f1']:.4f}"
    )
    print(
        f"ROC-AUC:           "
        f"{eeg_metrics['roc_auc']:.4f}"
    )
    print("Confusion:")
    print(
        eeg_metrics["confusion_matrix"]
    )

    print()
    print("fNIRS-only")
    print(
        f"Accuracy:          "
        f"{fnirs_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro Precision:   "
        f"{fnirs_metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall:      "
        f"{fnirs_metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1:          "
        f"{fnirs_metrics['macro_f1']:.4f}"
    )
    print(
        f"ROC-AUC:           "
        f"{fnirs_metrics['roc_auc']:.4f}"
    )
    print("Confusion:")
    print(
        fnirs_metrics["confusion_matrix"]
    )

    print()
    print("Fusion")
    print(
        f"Accuracy:          "
        f"{fusion_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro Precision:   "
        f"{fusion_metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall:      "
        f"{fusion_metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1:          "
        f"{fusion_metrics['macro_f1']:.4f}"
    )
    print(
        f"ROC-AUC:           "
        f"{fusion_metrics['roc_auc']:.4f}"
    )
    print("Confusion:")
    print(
        fusion_metrics["confusion_matrix"]
    )

    # -----------------------------------------------------
    # DIFFERENCES
    # -----------------------------------------------------

    eeg_f1 = eeg_metrics["macro_f1"]
    fnirs_f1 = fnirs_metrics["macro_f1"]
    fusion_f1 = fusion_metrics["macro_f1"]

    best_unimodal = max(
        eeg_f1,
        fnirs_f1
    )

    print()
    print(
        f"Fusion - EEG Macro-F1: "
        f"{fusion_f1 - eeg_f1:+.4f}"
    )

    print(
        f"Fusion - fNIRS Macro-F1: "
        f"{fusion_f1 - fnirs_f1:+.4f}"
    )

    print(
        f"Fusion - best unimodal: "
        f"{fusion_f1 - best_unimodal:+.4f}"
    )

    # -----------------------------------------------------
    # SAVE PREDICTIONS
    # -----------------------------------------------------

    prediction_path = os.path.join(
        PREDICTION_DIR,
        f"{test_subject}_predictions.npz"
    )

    np.savez_compressed(
        prediction_path,
        y_true=y_test,
        y_pred_eeg=eeg_pred,
        y_pred_fnirs=fnirs_pred,
        y_pred_fusion=fusion_pred,
        prob_eeg=eeg_prob,
        prob_fnirs=fnirs_prob,
        prob_fusion=fusion_prob,
        trial_id=trial_id_test,
        subject_id=subject_test,
        test_subject=test_subject
    )

    # -----------------------------------------------------
    # SAVE FOLD CHECKPOINT
    # -----------------------------------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{test_subject}_fold.pt"
    )

    torch.save(
        {
            "test_subject": test_subject,
            "validation_subject": val_subject,
            "train_subjects": train_subjects,

            "eeg_encoder_state_dict":
                eeg_encoder.state_dict(),

            "fnirs_encoder_state_dict":
                fnirs_encoder.state_dict(),

            "fusion_state_dict":
                fusion_model.state_dict(),

            "eeg_mean": eeg_mean,
            "eeg_std": eeg_std,

            "fnirs_mean": fnirs_mean,
            "fnirs_std": fnirs_std,

            "eeg_val_f1": eeg_val_f1,
            "fnirs_val_f1": fnirs_val_f1,
            "fusion_val_f1": fusion_val_f1,

            "seed": SEED
        },
        checkpoint_path
    )

    # -----------------------------------------------------
    # RETURN FOLD RESULTS
    # -----------------------------------------------------

    result = {
        "fold": fold_index + 1,
        "test_subject": test_subject,
        "validation_subject": val_subject,

        "eeg_accuracy":
            eeg_metrics["accuracy"],

        "eeg_macro_precision":
            eeg_metrics["macro_precision"],

        "eeg_macro_recall":
            eeg_metrics["macro_recall"],

        "eeg_macro_f1":
            eeg_metrics["macro_f1"],

        "eeg_roc_auc":
            eeg_metrics["roc_auc"],

        "fnirs_accuracy":
            fnirs_metrics["accuracy"],

        "fnirs_macro_precision":
            fnirs_metrics["macro_precision"],

        "fnirs_macro_recall":
            fnirs_metrics["macro_recall"],

        "fnirs_macro_f1":
            fnirs_metrics["macro_f1"],

        "fnirs_roc_auc":
            fnirs_metrics["roc_auc"],

        "fusion_accuracy":
            fusion_metrics["accuracy"],

        "fusion_macro_precision":
            fusion_metrics["macro_precision"],

        "fusion_macro_recall":
            fusion_metrics["macro_recall"],

        "fusion_macro_f1":
            fusion_metrics["macro_f1"],

        "fusion_roc_auc":
            fusion_metrics["roc_auc"],

        "fusion_minus_eeg_f1":
            fusion_f1 - eeg_f1,

        "fusion_minus_fnirs_f1":
            fusion_f1 - fnirs_f1,

        "fusion_minus_best_unimodal_f1":
            fusion_f1 - best_unimodal,

        "eeg_val_f1":
            eeg_val_f1,

        "fnirs_val_f1":
            fnirs_val_f1,

        "fusion_val_f1":
            fusion_val_f1
    }

    # Confusion matrices saved separately
    np.save(
        os.path.join(
            RESULTS_DIR,
            f"{test_subject}_confusion_eeg.npy"
        ),
        eeg_metrics["confusion_matrix"]
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            f"{test_subject}_confusion_fnirs.npy"
        ),
        fnirs_metrics["confusion_matrix"]
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            f"{test_subject}_confusion_fusion.npy"
        ),
        fusion_metrics["confusion_matrix"]
    )

    print()
    print(
        f"Saved predictions: {prediction_path}"
    )

    print(
        f"Saved checkpoint:  {checkpoint_path}"
    )

    return result


# ---------------------------------------------------------
# POOLED PREDICTIONS
# ---------------------------------------------------------

def create_pooled_predictions():

    records = []

    for subject in ALL_SUBJECTS:

        path = os.path.join(
            PREDICTION_DIR,
            f"{subject}_predictions.npz"
        )

        if not os.path.exists(path):
            continue

        data = np.load(
            path,
            allow_pickle=True
        )

        y_true = data["y_true"]
        y_eeg = data["y_pred_eeg"]
        y_fnirs = data["y_pred_fnirs"]
        y_fusion = data["y_pred_fusion"]

        p_eeg = data["prob_eeg"]
        p_fnirs = data["prob_fnirs"]
        p_fusion = data["prob_fusion"]

        trial_id = data["trial_id"]

        for i in range(
            len(y_true)
        ):

            row = {
                "subject": subject,
                "trial_id": int(trial_id[i]),
                "y_true": int(y_true[i]),

                "y_pred_eeg":
                    int(y_eeg[i]),

                "y_pred_fnirs":
                    int(y_fnirs[i]),

                "y_pred_fusion":
                    int(y_fusion[i]),

                "eeg_prob_0":
                    float(p_eeg[i, 0]),

                "eeg_prob_1":
                    float(p_eeg[i, 1]),

                "eeg_prob_2":
                    float(p_eeg[i, 2]),

                "fnirs_prob_0":
                    float(p_fnirs[i, 0]),

                "fnirs_prob_1":
                    float(p_fnirs[i, 1]),

                "fnirs_prob_2":
                    float(p_fnirs[i, 2]),

                "fusion_prob_0":
                    float(p_fusion[i, 0]),

                "fusion_prob_1":
                    float(p_fusion[i, 1]),

                "fusion_prob_2":
                    float(p_fusion[i, 2]),
            }

            records.append(row)

    if records:

        df = pd.DataFrame(
            records
        )

        output_path = os.path.join(
            RESULTS_DIR,
            "all_predictions.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        print()
        print(
            f"Saved pooled predictions: "
            f"{output_path}"
        )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

def create_summary(results):

    df = pd.DataFrame(
        results
    )

    per_subject_path = os.path.join(
        RESULTS_DIR,
        "per_subject_metrics.csv"
    )

    df.to_csv(
        per_subject_path,
        index=False
    )

    # -----------------------------------------------------
    # METRIC SUMMARY
    # -----------------------------------------------------

    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "roc_auc"
    ]

    summary_rows = []

    for metric in metrics:

        for modality in [
            "eeg",
            "fnirs",
            "fusion"
        ]:

            column = (
                f"{modality}_{metric}"
            )

            values = df[column].dropna()

            summary_rows.append(
                {
                    "modality": modality,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(
                        ddof=1
                    ),
                    "median": values.median(),
                    "min": values.min(),
                    "max": values.max()
                }
            )

    # -----------------------------------------------------
    # FUSION IMPROVEMENT
    # -----------------------------------------------------

    difference_columns = [
        "fusion_minus_eeg_f1",
        "fusion_minus_fnirs_f1",
        "fusion_minus_best_unimodal_f1"
    ]

    for column in difference_columns:

        values = df[column].dropna()

        summary_rows.append(
            {
                "modality": "fusion_difference",
                "metric": column,
                "mean": values.mean(),
                "std": values.std(
                    ddof=1
                ),
                "median": values.median(),
                "min": values.min(),
                "max": values.max()
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "loso_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # -----------------------------------------------------
    # PRINT FINAL SUMMARY
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL 24-FOLD LOSO SUMMARY")
    print("=" * 70)

    for modality in [
        "eeg",
        "fnirs",
        "fusion"
    ]:

        f1_values = df[
            f"{modality}_macro_f1"
        ]

        auc_values = df[
            f"{modality}_roc_auc"
        ].dropna()

        print()
        print(
            modality.upper()
        )

        print(
            f"Macro-F1: "
            f"{f1_values.mean():.4f} "
            f"+/- "
            f"{f1_values.std(ddof=1):.4f}"
        )

        print(
            f"ROC-AUC:  "
            f"{auc_values.mean():.4f} "
            f"+/- "
            f"{auc_values.std(ddof=1):.4f}"
        )

    print()
    print(
        "Fusion - EEG Macro-F1:"
    )

    print(
        f"{df['fusion_minus_eeg_f1'].mean():+.4f} "
        f"+/- "
        f"{df['fusion_minus_eeg_f1'].std(ddof=1):.4f}"
    )

    print()
    print(
        "Fusion - fNIRS Macro-F1:"
    )

    print(
        f"{df['fusion_minus_fnirs_f1'].mean():+.4f} "
        f"+/- "
        f"{df['fusion_minus_fnirs_f1'].std(ddof=1):.4f}"
    )

    print()
    print(
        "Fusion - best unimodal Macro-F1:"
    )

    print(
        f"{df['fusion_minus_best_unimodal_f1'].mean():+.4f} "
        f"+/- "
        f"{df['fusion_minus_best_unimodal_f1'].std(ddof=1):.4f}"
    )

    print()
    print(
        f"Saved per-subject metrics: "
        f"{per_subject_path}"
    )

    print(
        f"Saved summary: "
        f"{summary_path}"
    )

    return df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="24-fold LOSO EEG + fNIRS experiment"
    )

    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
        help="Run only one LOSO fold, e.g. VP001"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 24 LOSO folds"
    )

    args = parser.parse_args()

    set_seed(SEED)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        PREDICTION_DIR,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("EEG + fNIRS LOSO EXPERIMENT")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Subjects: {len(ALL_SUBJECTS)}"
    )

    print(
        f"EEG epochs: {EEG_EPOCHS}"
    )

    print(
        f"fNIRS epochs: {FNIRS_EPOCHS}"
    )

    print(
        f"Fusion epochs: {FUSION_EPOCHS}"
    )

    print(
        "Test subjects are completely held out per fold."
    )

    # -----------------------------------------------------
    # SELECT FOLDS
    # -----------------------------------------------------

    if args.test_subject:

        if args.test_subject not in ALL_SUBJECTS:

            raise ValueError(
                f"Unknown subject: "
                f"{args.test_subject}"
            )

        subjects_to_run = [
            args.test_subject
        ]

    elif args.all:

        subjects_to_run = ALL_SUBJECTS

    else:

        print()
        print(
            "Specify either:"
        )

        print(
            "  --test-subject VP001"
        )

        print(
            "or:"
        )

        print(
            "  --all"
        )

        return

    # -----------------------------------------------------
    # RUN FOLDS
    # -----------------------------------------------------

    results = []

    for subject in subjects_to_run:

        fold_index = ALL_SUBJECTS.index(
            subject
        )

        try:

            result = run_fold(
                subject,
                fold_index
            )

            results.append(
                result
            )

        except Exception as e:

            print()
            print(
                f"ERROR in fold {subject}:"
            )

            print(
                repr(e)
            )

            raise

        # Save progress after every fold
        if results:

            progress_df = pd.DataFrame(
                results
            )

            progress_path = os.path.join(
                RESULTS_DIR,
                "loso_progress.csv"
            )

            progress_df.to_csv(
                progress_path,
                index=False
            )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    if results:

        create_summary(
            results
        )

        create_pooled_predictions()

    print()
    print("=" * 70)

    if len(results) == len(
        subjects_to_run
    ):

        print(
            "LOSО RUN COMPLETE"
        )

    else:

        print(
            "LOSО RUN PARTIALLY COMPLETE"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()