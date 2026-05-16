import os
import csv
from datetime import datetime


# CSV HEADER
CSV_COLUMNS = [
    "experiment",
    "stage",
    "epoch",

    "train_loss",
    "train_acc",
    "train_f1",
    "train_recall",

    "val_loss",
    "val_acc",
    "val_f1",
    "val_recall",

    "test_accuracy",
    "test_f1",
    "test_recall",

    "status",
]


# LOGGER FUNCTION
def log_results(cfg, metrics: dict):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "results.csv")
    txt_path = os.path.join(results_dir, f"{cfg.experiment}_log.txt")




    # PREPARE CSV ROW

    row = []

    for col in CSV_COLUMNS:
        value = metrics.get(col, "")

        if value is None:
            value = ""

        row.append(value)


    # WRITE CSV

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_COLUMNS)

        writer.writerow(row)


    # WRITE TXT LOG (IMPROVED)

    with open(txt_path, "a", encoding="utf-8") as f:

        f.write("\n*******************************\n")
        f.write(f"Experiment: {cfg.experiment}\n")
        f.write(f"Stage: {metrics.get('stage')}\n")
        f.write(f"Epoch: {metrics.get('epoch')}\n")

        # TRAIN
        if metrics.get("train_loss") != "":
            f.write(
                f"\nTrain:\n"
                f"  Loss: {metrics.get('train_loss')}\n"
                f"  Acc: {metrics.get('train_acc')}\n"
                f"  F1: {metrics.get('train_f1')}\n"
                f"  Recall: {metrics.get('train_recall')}\n"
            )

        # VALIDATION
        if metrics.get("val_loss") != "":
            f.write(
                f"\nValidation:\n"
                f"  Loss: {metrics.get('val_loss')}\n"
                f"  Acc: {metrics.get('val_acc')}\n"
                f"  F1: {metrics.get('val_f1')}\n"
                f"  Recall: {metrics.get('val_recall')}\n"
            )

        # TEST
        if metrics.get("test_accuracy") != "":
            f.write(
                f"\nTest:\n"
                f"  Accuracy: {metrics.get('test_accuracy')}\n"
                f"  F1: {metrics.get('test_f1')}\n"
                f"  Recall: {metrics.get('test_recall')}\n"
            )

        f.write(f"\nStatus: {metrics.get('status')}\n")
        f.write("*******************************\n")

    print(" Logged:", row)