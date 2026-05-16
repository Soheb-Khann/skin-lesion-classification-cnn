import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize



# BASE RESULTS DIR

def get_results_dir():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



# CONFUSION MATRIX

def save_confusion_matrix(y_true, y_pred, cfg):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{cfg.experiment} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    path = os.path.join(get_results_dir(), f"{cfg.experiment}_cm.png")
    plt.savefig(path)
    plt.close()

    print(" Confusion Matrix saved")



# ROC CURVE (MULTI-CLASS)

def save_roc_curve(y_true, y_probs, cfg, num_classes=7):

    try:
        y_probs = np.array(y_probs)

        # Safety check
        if len(y_probs.shape) != 2:
            raise ValueError("y_probs is not 2D")

        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        plt.figure(figsize=(8, 6))

        # Store AUCs for macro
        aucs = []

        for i in range(num_classes):

            if np.sum(y_true_bin[:, i]) == 0:
                continue

            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)

            aucs.append(roc_auc)

            plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

        # Macro AUC
        if len(aucs) > 0:
            macro_auc = np.mean(aucs)
            plt.plot([], [], ' ', label=f"Macro AUC = {macro_auc:.2f}")

        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.title(f"{cfg.experiment} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        plt.tight_layout()

        path = os.path.join(get_results_dir(), f"{cfg.experiment}_roc.png")
        plt.savefig(path)
        plt.close()

        print(" ROC Curve saved")

    except Exception as e:
        print(" ROC Curve failed:", e)



# PRECISION-RECALL CURVE

def save_pr_curve(y_true, y_probs, cfg, num_classes=7):

    try:
        y_probs = np.array(y_probs)

        if len(y_probs.shape) != 2:
            raise ValueError("y_probs is not 2D")

        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        plt.figure(figsize=(8, 6))

        pr_aucs = []

        for i in range(num_classes):

            if np.sum(y_true_bin[:, i]) == 0:
                continue

            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_probs[:, i]
            )

            pr_auc = auc(recall, precision)
            pr_aucs.append(pr_auc)

            plt.plot(recall, precision, label=f"Class {i} (AUC={pr_auc:.2f})")

        # Macro PR AUC
        if len(pr_aucs) > 0:
            macro_pr_auc = np.mean(pr_aucs)
            plt.plot([], [], ' ', label=f"Macro PR AUC = {macro_pr_auc:.2f}")

        plt.title(f"{cfg.experiment} - Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        plt.tight_layout()

        path = os.path.join(get_results_dir(), f"{cfg.experiment}_pr.png")
        plt.savefig(path)
        plt.close()

        print(" Precision-Recall Curve saved")

    except Exception as e:
        print(" PR Curve failed:", e)