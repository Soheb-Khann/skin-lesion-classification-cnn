import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

from training.metrics import (
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve
)
from utils.logger import log_results


def train_model(model, train_loader, val_loader, train_df, cfg):

    device = cfg.device
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)


    # LOSS FUNCTION

    if cfg.use_weighted_loss:

        class_counts = np.bincount(train_df["label"])

        weights = 1.0 / np.sqrt(class_counts)
        weights = weights / weights.sum()

        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        print("\nClass counts:", class_counts)
        print("Weights:", weights)

        loss_fn = nn.CrossEntropyLoss(weight=weights)

    else:
        loss_fn = nn.CrossEntropyLoss()


    # EARLY STOPPING CONFIG

    best_f1 = 0
    patience = 3
    counter = 0
    min_delta = 0.003

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "results", f"{cfg.experiment}_model.pth")

    for epoch in range(cfg.epochs):


        # TRAINING

        model.train()

        train_losses = []
        train_preds = []
        train_labels = []

        for imgs, feats, labels in tqdm(train_loader):

            imgs = imgs.to(device)
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs, feats)
            loss = loss_fn(outputs, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro")
        train_recall = recall_score(train_labels, train_preds, average="macro")


        # VALIDATION

        model.eval()

        val_losses = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for imgs, feats, labels in val_loader:

                imgs = imgs.to(device)
                feats = feats.to(device)
                labels = labels.to(device)

                outputs = model(imgs, feats)
                loss = loss_fn(outputs, labels)

                val_losses.append(loss.item())

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_recall = recall_score(val_labels, val_preds, average="macro")


        # FIT STATUS DETECTION

        if train_f1 < val_f1:
            fit_status = "UNDERFITTING"
        elif (train_f1 - val_f1) > 0.15:
            fit_status = "OVERFITTING"
        else:
            fit_status = "GOOD FIT"

        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        print(f"Status: {fit_status}")


        # EARLY STOPPING LOGIC

        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1
            counter = 0

            torch.save(model.state_dict(), model_path)
            print(" Model saved")

            status = "BEST"

        else:
            counter += 1
            status = "NORMAL"

        if counter >= patience:
            print(" Early stopping triggered")
            break


        # LOGGING

        metrics = {
            "experiment": cfg.experiment,
            "stage": "train",
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "train_f1": round(train_f1, 4),
            "train_recall": round(train_recall, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_recall": round(val_recall, 4),
            "test_accuracy": "",
            "test_f1": "",
            "test_recall": "",
            "status": fit_status
        }

        log_results(cfg, metrics)



# TEST EVALUATION


def evaluate_on_test(model, test_loader, device, cfg):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, feats, labels in test_loader:

            imgs = imgs.to(device)
            feats = feats.to(device)

            outputs = model(imgs, feats)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs_np = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Macro Recall: {recall:.4f}")

    report = classification_report(all_labels, all_preds)
    print("\nClassification Report:\n", report)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    report_path = os.path.join(BASE_DIR, "results", f"{cfg.experiment}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    save_confusion_matrix(all_labels, all_preds, cfg)
    save_roc_curve(all_labels, all_probs_np, cfg)
    save_pr_curve(all_labels, all_probs_np, cfg)

    metrics = {
        "experiment": cfg.experiment,
        "stage": "test",
        "epoch": "final",
        "train_loss": "",
        "train_acc": "",
        "train_f1": "",
        "train_recall": "",
        "val_loss": "",
        "val_acc": "",
        "val_f1": "",
        "val_recall": "",
        "test_accuracy": round(acc, 4),
        "test_f1": round(f1, 4),
        "test_recall": round(recall, 4),
        "status": "FINAL"
    }

    log_results(cfg, metrics)

    return acc, f1, recall