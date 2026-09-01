from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["NEGATIVE", "POSITIVE"],
            output_dict=True,
            zero_division=0
        )
    }


def save_confusion_matrix(y_true, y_pred, model_name):
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NEGATIVE", "POSITIVE"],
        yticklabels=["NEGATIVE", "POSITIVE"]
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    image_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(RESULTS_DIR / image_name, dpi=160)
    plt.close()


def save_comparison_chart(metrics_by_model):
    model_names = list(metrics_by_model.keys())
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    positions = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(9, 5))

    for index, model_name in enumerate(model_names):
        values = [metrics_by_model[model_name][metric] for metric in metric_names]
        offset = (index - (len(model_names) - 1) / 2) * width
        plt.bar(positions + offset, values, width, label=model_name)

    plt.xticks(positions, ["Accuracy", "Precision", "Recall", "F1-score"])
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Naive Bayes vs Logistic Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=160)
    plt.close()
