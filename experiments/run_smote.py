import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import CFG, update_cfg, print_pipeline
from data.dataset import prepare_dataframe
from data.split import split_dataset
from training.dataloader import get_dataloaders
from models.hybrid_model import HybridModel
from training.trainer import train_model
from training.smote_pipeline import (
    extract_features,
    apply_smote,
    train_smote_classifier,
    evaluate_smote
)

from utils.seed import set_seed

import torch


if __name__ == "__main__":

    set_seed(42)

    CFG.experiment = "smote"
    CFG = update_cfg(CFG)
    print_pipeline(CFG)

    # OPTIONAL SAFETY
    if not CFG.use_smote:
        raise ValueError("SMOTE flag not enabled properly")

    # DATA
    df, _ = prepare_dataframe(CFG)
    train_df, val_df, test_df = split_dataset(df)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, CFG
    )

    # STEP 1: TRAIN BASE MODEL
    model = HybridModel(CFG).to(CFG.device)

    train_model(model, train_loader, val_loader, train_df, CFG)

    #  LOAD BEST MODEL
    model_path = os.path.join(PROJECT_ROOT, "results", f"{CFG.experiment}_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=CFG.device))
        print(" Loaded best model for SMOTE feature extraction")

    # STEP 2: EXTRACT FEATURES
    X, y = extract_features(model, train_loader, CFG.device)

    # STEP 3: APPLY SMOTE
    X_res, y_res, scaler = apply_smote(X, y)

    # STEP 4: TRAIN SMOTE CLASSIFIER
    smote_model = train_smote_classifier(X_res, y_res, CFG)

    print("\n SMOTE feature extraction + training completed")

    # STEP 5: EVALUATE
    evaluate_smote(smote_model, model, test_loader, CFG, scaler=scaler)
