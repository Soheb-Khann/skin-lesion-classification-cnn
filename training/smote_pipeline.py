import torch
import numpy as np
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import os

import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler

from training.metrics import (
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve
)
from utils.logger import log_results



# FEATURE EXTRACTION


def extract_features(model, loader, device):

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for imgs, feats, lbls in tqdm(loader):

            imgs = imgs.to(device)
            feats = feats.to(device)

            #  USE PROPER FEATURE EXTRACTOR
            combined = model.extract_features(imgs, feats)

            features.append(combined.cpu().numpy())
            labels.append(lbls.cpu().numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    print(" Feature extraction complete:", features.shape)
    print(" Using tabular:", model.use_tabular)

    return features, labels



# APPLY SMOTE


def apply_smote(features, labels):

    #  SCALE FEATURES
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    smote = SMOTE(random_state=42)

    X_res, y_res = smote.fit_resample(features, labels)

    print("\n SMOTE applied:")
    print(" Before:", np.bincount(labels))
    print(" After :", np.bincount(y_res))

    return X_res, y_res, scaler



# CLASSIFIER


class SMOTEClassifier(nn.Module):

    def __init__(self, input_dim, num_classes=7):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)



# TRAIN CLASSIFIER


def train_smote_classifier(X, y, cfg):

    device = cfg.device

    #  SHUFFLE DATA
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    model = SMOTEClassifier(X.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 30

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(X)
        loss = loss_fn(outputs, y)

        loss.backward()

        #  GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

    return model



# EVALUATE SMOTE MODEL


def evaluate_smote(model, feature_extractor, test_loader, cfg, scaler=None):

    device = cfg.device
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    # LOG FILE
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(BASE_DIR, "results", f"{cfg.experiment}_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")

    with torch.no_grad():
        for imgs, feats, labels in test_loader:

            imgs = imgs.to(device)
            feats = feats.to(device)

            # FEATURE EXTRACTION
            combined = feature_extractor.extract_features(imgs, feats)

            if scaler is not None:
                combined_np = scaler.transform(combined.cpu().numpy())
                combined = torch.tensor(combined_np, dtype=torch.float32, device=device)

            outputs = model(combined)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs_np = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")

    print("\n===== SMOTE TEST RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Macro Recall: {recall:.4f}")

    report = classification_report(all_labels, all_preds)
    print("\nClassification Report:\n", report)

    # SAVE REPORT
    report_path = os.path.join(BASE_DIR, "results", f"{cfg.experiment}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # LOG FILE
    log_file.write("\n===== TEST RESULTS =====\n")
    log_file.write(f"Accuracy: {acc:.4f}\n")
    log_file.write(f"Macro F1: {f1:.4f}\n")
    log_file.write(f"Macro Recall: {recall:.4f}\n")
    log_file.write("\nClassification Report:\n")
    log_file.write(report)
    log_file.close()

    # VISUALS
    save_confusion_matrix(all_labels, all_preds, cfg)
    save_roc_curve(all_labels, all_probs_np, cfg)
    save_pr_curve(all_labels, all_probs_np, cfg)

    # CSV LOG
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
