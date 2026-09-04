from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "results"
    / "loso"
    / "predictions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "loso"
    / "analysis"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

CLASSES = [0, 1, 2]

CLASS_NAMES = [
    "0-back",
    "2-back",
    "3-back",
]

MODELS = [
    "EEG",
    "fNIRS",
    "Fusion",
]


# ============================================================
# ACTUAL NPZ FIELD NAMES FROM YOUR LOSO SCRIPT
# ============================================================

PREDICTION_FIELDS = {
    "EEG": "y_pred_eeg",
    "fNIRS": "y_pred_fnirs",
    "Fusion": "y_pred_fusion",
}

PROBABILITY_FIELDS = {
    "EEG": "prob_eeg",
    "fNIRS": "prob_fnirs",
    "Fusion": "prob_fusion",
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    probabilities
):

    result = {}

    result["accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    result["macro_precision"] = precision_score(
        y_true,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    result["macro_recall"] = recall_score(
        y_true,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    result["macro_f1"] = f1_score(
        y_true,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    if len(np.unique(y_true)) == 3:

        try:

            result["roc_auc"] = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                labels=CLASSES
            )

        except Exception:

            result["roc_auc"] = np.nan

    else:

        result["roc_auc"] = np.nan

    return result


# ============================================================
# VALIDATE ONE NPZ
# ============================================================

def validate_file(
    subject,
    file_path
):

    errors = []

    try:

        data = np.load(
            file_path,
            allow_pickle=True
        )

    except Exception as e:

        return [], [
            f"Could not load file: {e}"
        ]

    required_fields = [
        "y_true",
        "y_pred_eeg",
        "y_pred_fnirs",
        "y_pred_fusion",
        "prob_eeg",
        "prob_fnirs",
        "prob_fusion",
        "trial_id",
        "subject_id",
        "test_subject",
    ]

    for field in required_fields:

        if field not in data.files:

            errors.append(
                f"Missing field: {field}"
            )

    if errors:

        return [], errors

    y_true = np.asarray(
        data["y_true"]
    ).reshape(-1).astype(int)

    n = len(y_true)

    # --------------------------------------------------------
    # Basic y_true validation
    # --------------------------------------------------------

    if n != 27:

        errors.append(
            f"Expected 27 epochs, found {n}"
        )

    if not np.all(
        np.isfinite(y_true)
    ):

        errors.append(
            "y_true contains NaN/Inf"
        )

    invalid_true = (
        set(np.unique(y_true))
        - set(CLASSES)
    )

    if invalid_true:

        errors.append(
            f"Invalid true labels: {invalid_true}"
        )

    records = []

    # --------------------------------------------------------
    # Validate each model
    # --------------------------------------------------------

    for model in MODELS:

        pred_field = PREDICTION_FIELDS[
            model
        ]

        prob_field = PROBABILITY_FIELDS[
            model
        ]

        y_pred = np.asarray(
            data[pred_field]
        ).reshape(-1).astype(int)

        probabilities = np.asarray(
            data[prob_field]
        )

        # Prediction length
        if len(y_pred) != n:

            errors.append(
                f"{model}: prediction length "
                f"{len(y_pred)} != {n}"
            )

        # Probability shape
        if probabilities.shape != (
            n,
            3
        ):

            errors.append(
                f"{model}: probability shape "
                f"{probabilities.shape}; "
                f"expected ({n}, 3)"
            )

        # Probability finite
        if not np.all(
            np.isfinite(probabilities)
        ):

            errors.append(
                f"{model}: probabilities contain NaN/Inf"
            )

        # Probability range
        if np.any(
            probabilities < -1e-5
        ):

            errors.append(
                f"{model}: negative probabilities"
            )

        if np.any(
            probabilities > 1 + 1e-5
        ):

            errors.append(
                f"{model}: probability > 1"
            )

        # Probability sum
        if probabilities.ndim == 2:

            if not np.allclose(
                probabilities.sum(axis=1),
                1.0,
                atol=1e-4
            ):

                errors.append(
                    f"{model}: probability rows "
                    f"do not sum to 1"
                )

        # Prediction labels
        invalid_pred = (
            set(np.unique(y_pred))
            - set(CLASSES)
        )

        if invalid_pred:

            errors.append(
                f"{model}: invalid predictions "
                f"{invalid_pred}"
            )

        # Prediction should match probability argmax
        if probabilities.shape == (
            n,
            3
        ):

            argmax_pred = np.argmax(
                probabilities,
                axis=1
            )

            if not np.array_equal(
                y_pred,
                argmax_pred
            ):

                errors.append(
                    f"{model}: saved prediction "
                    f"does not match probability argmax"
                )

        metrics = calculate_metrics(
            y_true,
            y_pred,
            probabilities
        )

        records.append({
            "subject": subject,
            "model": model,
            **metrics
        })

    return records, errors


# ============================================================
# BAR GRAPH
# ============================================================

def make_metric_graph(
    summary,
    metric,
    title,
    filename
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    means = [
        summary.loc[
            model,
            f"Mean {metric}"
        ]
        for model in MODELS
    ]

    sds = [
        summary.loc[
            model,
            f"SD {metric}"
        ]
        for model in MODELS
    ]

    x = np.arange(
        len(MODELS)
    )

    ax.bar(
        x,
        means,
        yerr=sds,
        capsize=5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)

    ax.set_ylabel(metric)

    ax.set_title(title)

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# PER SUBJECT F1
# ============================================================

def make_per_subject_f1(
    df
):

    pivot = df.pivot(
        index="subject",
        columns="model",
        values="macro_f1"
    ).reindex(
        columns=MODELS
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    x = np.arange(
        len(pivot)
    )

    width = 0.25

    for i, model in enumerate(MODELS):

        ax.bar(
            x + (i - 1) * width,
            pivot[model].values,
            width,
            label=model
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        pivot.index,
        rotation=45
    )

    ax.set_ylabel(
        "Macro-F1"
    )

    ax.set_title(
        "Macro-F1 by held-out subject"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        "per_subject_macro_f1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return pivot


# ============================================================
# FUSION IMPROVEMENT
# ============================================================

def make_fusion_improvement(
    pivot
):

    comparison = pd.DataFrame(
        index=pivot.index
    )

    comparison[
        "Fusion_minus_EEG"
    ] = (
        pivot["Fusion"]
        - pivot["EEG"]
    )

    comparison[
        "Fusion_minus_fNIRS"
    ] = (
        pivot["Fusion"]
        - pivot["fNIRS"]
    )

    comparison[
        "Fusion_minus_best"
    ] = (
        pivot["Fusion"]
        - pivot[
            ["EEG", "fNIRS"]
        ].max(axis=1)
    )

    comparison[
        "Fusion_wins"
    ] = (
        comparison[
            "Fusion_minus_best"
        ] > 0
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    x = np.arange(
        len(comparison)
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.bar(
        x,
        comparison[
            "Fusion_minus_best"
        ]
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        comparison.index,
        rotation=45
    )

    ax.set_ylabel(
        "Macro-F1 difference"
    )

    ax.set_title(
        "Fusion − Best Unimodal Macro-F1"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        "fusion_improvement.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return comparison


# ============================================================
# CONFUSION MATRIX
# ============================================================

def make_confusion_matrix(
    y_true,
    y_pred,
    model
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(cm)

    ax.set_xticks(
        range(3)
    )

    ax.set_yticks(
        range(3)
    )

    ax.set_xticklabels(
        CLASS_NAMES
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        f"{model} pooled confusion matrix"
    )

    for i in range(3):

        for j in range(3):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    filename = (
        model.lower()
        + "_confusion_matrix.png"
    )

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return cm


# ============================================================
# FUSION CLASSWISE
# ============================================================

def make_classwise(
    y_true,
    y_pred
):

    precision = precision_score(
        y_true,
        y_pred,
        labels=CLASSES,
        average=None,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=CLASSES,
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASSES,
        average=None,
        zero_division=0
    )

    result = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    x = np.arange(3)

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width,
        precision,
        width,
        label="Precision"
    )

    ax.bar(
        x,
        recall,
        width,
        label="Recall"
    )

    ax.bar(
        x + width,
        f1,
        width,
        label="F1"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        CLASS_NAMES
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        "Fusion class-wise performance"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        "fusion_classwise_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return result


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

def make_prediction_distribution(
    y_true,
    y_pred
):

    actual = np.bincount(
        y_true,
        minlength=3
    )

    predicted = np.bincount(
        y_pred,
        minlength=3
    )

    x = np.arange(3)

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        actual,
        width,
        label="Actual"
    )

    ax.bar(
        x + width / 2,
        predicted,
        width,
        label="Predicted"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        CLASS_NAMES
    )

    ax.set_ylabel(
        "Number of epochs"
    )

    ax.set_title(
        "Fusion actual vs predicted distribution"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        "fusion_prediction_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("LOSO RESULTS VALIDATION + PPT ANALYSIS")
    print("=" * 70)

    print(
        f"Prediction folder:\n"
        f"{PREDICTION_DIR}"
    )

    files = sorted(
        PREDICTION_DIR.glob(
            "VP*_predictions.npz"
        )
    )

    print()
    print(
        f"Completed prediction files: "
        f"{len(files)} / 24"
    )

    if not files:

        print(
            "\nNo prediction files found."
        )

        return

    all_records = []

    pooled = {
        model: {
            "true": [],
            "pred": [],
            "prob": []
        }
        for model in MODELS
    }

    validation_errors = []

    # ========================================================
    # VALIDATE EACH FOLD
    # ========================================================

    for file_path in files:

        subject = (
            file_path.stem
            .replace(
                "_predictions",
                ""
            )
        )

        print(
            f"\nChecking {subject} ..."
        )

        records, errors = validate_file(
            subject,
            file_path
        )

        all_records.extend(
            records
        )

        if errors:

            print(
                f"  FAIL ({len(errors)} issue(s))"
            )

            for error in errors:

                print(
                    f"   - {error}"
                )

                validation_errors.append(
                    f"{subject}: {error}"
                )

        else:

            print(
                "  PASS"
            )

        # ----------------------------------------------------
        # Load pooled arrays
        # ----------------------------------------------------

        try:

            data = np.load(
                file_path,
                allow_pickle=True
            )

            y_true = np.asarray(
                data["y_true"]
            ).reshape(-1).astype(int)

            for model in MODELS:

                probs = np.asarray(
                    data[
                        PROBABILITY_FIELDS[
                            model
                        ]
                    ]
                )

                preds = np.asarray(
                    data[
                        PREDICTION_FIELDS[
                            model
                        ]
                    ]
                ).reshape(-1).astype(int)

                pooled[
                    model
                ]["true"].append(
                    y_true
                )

                pooled[
                    model
                ]["pred"].append(
                    preds
                )

                pooled[
                    model
                ]["prob"].append(
                    probs
                )

        except Exception as e:

            validation_errors.append(
                f"{subject}: pooled load error: {e}"
            )

    # ========================================================
    # METRICS TABLE
    # ========================================================

    df = pd.DataFrame(
        all_records
    )

    if df.empty:

        print(
            "\nNo valid results available."
        )

        return

    df.to_csv(
        OUTPUT_DIR /
        "recomputed_metrics.csv",
        index=False
    )

    # ========================================================
    # MEAN ± SD
    # ========================================================

    summary_rows = []

    for model in MODELS:

        model_df = df[
            df["model"] == model
        ]

        if model_df.empty:

            continue

        summary_rows.append({

            "Model": model,

            "N_subjects":
                model_df[
                    "subject"
                ].nunique(),

            "Mean Accuracy":
                model_df[
                    "accuracy"
                ].mean(),

            "SD Accuracy":
                model_df[
                    "accuracy"
                ].std(
                    ddof=1
                ),

            "Mean Macro Precision":
                model_df[
                    "macro_precision"
                ].mean(),

            "SD Macro Precision":
                model_df[
                    "macro_precision"
                ].std(
                    ddof=1
                ),

            "Mean Macro Recall":
                model_df[
                    "macro_recall"
                ].mean(),

            "SD Macro Recall":
                model_df[
                    "macro_recall"
                ].std(
                    ddof=1
                ),

            "Mean Macro F1":
                model_df[
                    "macro_f1"
                ].mean(),

            "SD Macro F1":
                model_df[
                    "macro_f1"
                ].std(
                    ddof=1
                ),

            "Mean ROC-AUC":
                model_df[
                    "roc_auc"
                ].mean(),

            "SD ROC-AUC":
                model_df[
                    "roc_auc"
                ].std(
                    ddof=1
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_DIR /
        "summary_mean_sd.csv",
        index=False
    )

    print()
    print("=" * 70)
    print("MEAN ± SD ACROSS COMPLETED FOLDS")
    print("=" * 70)

    print(
        summary.round(4)
        .to_string(index=False)
    )

    # ========================================================
    # FUSION COMPARISON
    # ========================================================

    pivot = df.pivot(
        index="subject",
        columns="model",
        values="macro_f1"
    ).reindex(
        columns=MODELS
    )

    if all(
        model in pivot.columns
        for model in MODELS
    ):

        pivot[
            "Fusion_minus_EEG"
        ] = (
            pivot["Fusion"]
            - pivot["EEG"]
        )

        pivot[
            "Fusion_minus_fNIRS"
        ] = (
            pivot["Fusion"]
            - pivot["fNIRS"]
        )

        pivot[
            "Fusion_minus_best"
        ] = (
            pivot["Fusion"]
            - pivot[
                ["EEG", "fNIRS"]
            ].max(axis=1)
        )

        pivot[
            "Fusion_wins"
        ] = (
            pivot[
                "Fusion_minus_best"
            ] > 0
        )

        pivot.to_csv(
            OUTPUT_DIR /
            "fusion_comparison_by_subject.csv"
        )

        print()
        print("=" * 70)
        print("FUSION COMPARISON")
        print("=" * 70)

        print(
            f"Fusion - EEG Macro-F1: "
            f"{pivot['Fusion_minus_EEG'].mean():+.4f}"
        )

        print(
            f"Fusion - fNIRS Macro-F1: "
            f"{pivot['Fusion_minus_fNIRS'].mean():+.4f}"
        )

        print(
            f"Fusion - best unimodal: "
            f"{pivot['Fusion_minus_best'].mean():+.4f}"
        )

        print(
            f"Fusion wins: "
            f"{int(pivot['Fusion_wins'].sum())}"
            f"/{len(pivot)} subjects"
        )

        make_per_subject_f1(
            df
        )

        make_fusion_improvement(
            pivot
        )

    # ========================================================
    # POOLED RESULTS
    # ========================================================

    pooled_rows = []

    for model in MODELS:

        if not pooled[
            model
        ]["true"]:

            continue

        y_true = np.concatenate(
            pooled[model]["true"]
        )

        y_pred = np.concatenate(
            pooled[model]["pred"]
        )

        probs = np.concatenate(
            pooled[model]["prob"]
        )

        # Save pooled prediction table
        for i in range(
            len(y_true)
        ):

            pooled_rows.append({

                "model": model,

                "y_true":
                    int(y_true[i]),

                "y_pred":
                    int(y_pred[i]),

                "prob_0back":
                    float(probs[i, 0]),

                "prob_2back":
                    float(probs[i, 1]),

                "prob_3back":
                    float(probs[i, 2]),
            })

        make_confusion_matrix(
            y_true,
            y_pred,
            model
        )

        if model == "Fusion":

            classwise = (
                make_classwise(
                    y_true,
                    y_pred
                )
            )

            classwise.to_csv(
                OUTPUT_DIR /
                "classwise_metrics.csv",
                index=False
            )

            make_prediction_distribution(
                y_true,
                y_pred
            )

    if pooled_rows:

        pd.DataFrame(
            pooled_rows
        ).to_csv(
            OUTPUT_DIR /
            "pooled_predictions.csv",
            index=False
        )

    # ========================================================
    # MAIN COMPARISON GRAPHS
    # ========================================================

    summary_indexed = (
        summary.set_index(
            "Model"
        )
    )

    if all(
        model in summary_indexed.index
        for model in MODELS
    ):

        make_metric_graph(
            summary_indexed,
            "Accuracy",
            "Mean Accuracy Across Completed LOSO Subjects",
            "mean_accuracy.png"
        )

        make_metric_graph(
            summary_indexed,
            "Macro F1",
            "Mean Macro-F1 Across Completed LOSO Subjects",
            "mean_macro_f1.png"
        )

        make_metric_graph(
            summary_indexed,
            "ROC-AUC",
            "Mean ROC-AUC Across Completed LOSO Subjects",
            "mean_roc_auc.png"
        )

    # ========================================================
    # VALIDATION REPORT
    # ========================================================

    report = []

    report.append(
        "LOSO RESULTS VALIDATION REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append(
        f"Prediction files found: "
        f"{len(files)} / 24"
    )

    report.append(
        f"Validation errors: "
        f"{len(validation_errors)}"
    )

    report.append("")

    if validation_errors:

        report.append(
            "VERDICT: CHECK ERRORS"
        )

        report.append("")

        for error in validation_errors:

            report.append(
                error
            )

    else:

        report.append(
            "VERDICT: PASS"
        )

    report.append("")

    report.append(
        "Completed subjects:"
    )

    report.append(
        ", ".join(
            subject.replace(
                "_predictions",
                ""
            )
            for subject in [
                f.stem
                for f in files
            ]
        )
    )

    report.append("")

    report.append(
        "These are preliminary LOSO results "
        "until all 24 subjects are completed."
    )

    with open(
        OUTPUT_DIR /
        "validation_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # ========================================================
    # FINAL SCREEN
    # ========================================================

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    if validation_errors:

        print(
            f"VALIDATION: "
            f"{len(validation_errors)} ISSUE(S)"
        )

    else:

        print(
            "VALIDATION: PASS"
        )

    print()
    print(
        f"Folds analyzed: "
        f"{len(files)} / 24"
    )

    print()
    print(
        "Analysis saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Figures saved to:"
    )

    print(
        FIGURE_DIR
    )


if __name__ == "__main__":

    main()