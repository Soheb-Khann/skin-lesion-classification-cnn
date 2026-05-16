import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CFG, update_cfg, print_pipeline
from data.dataset import prepare_dataframe
from data.split import split_dataset
from training.dataloader import get_dataloaders
from models.hybrid_model import HybridModel

from training.smote_pipeline import (
    extract_features,
    apply_smote,
    train_smote_classifier,
    evaluate_smote
)

from training.trainer import train_model
from utils.seed import set_seed

import torch


if __name__ == "__main__":

    # SEED
    set_seed(42)

    print("\n Running CONTRAST + HYBRID + SMOTE experiment")

    # CONFIG
    CFG.experiment = "contrast_hybrid_smote"
    CFG = update_cfg(CFG)

    print_pipeline(CFG)

    # DATA
    df, _ = prepare_dataframe(CFG)

    train_df, val_df, test_df = split_dataset(df)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, CFG
    )

    # STEP 1: FEATURE EXTRACTOR TRAINING
    print("\n Training feature extractor (HybridModel)...")

    feature_extractor = HybridModel(CFG).to(CFG.device)

    train_model(feature_extractor, train_loader, val_loader, train_df, CFG)

    # LOAD BEST FEATURE EXTRACTOR
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "results", f"{CFG.experiment}_model.pth")

    if os.path.exists(model_path):
        feature_extractor.load_state_dict(
            torch.load(model_path, map_location=CFG.device)
        )
        print("\n Loaded best feature extractor")

    # STEP 2: FEATURE EXTRACTION
    print("\n Extracting features...")

    X_train, y_train = extract_features(
        feature_extractor, train_loader, CFG.device
    )

    # STEP 3: APPLY SMOTE
    print("\n Applying SMOTE...")

    X_resampled, y_resampled, scaler = apply_smote(X_train, y_train)

    # STEP 4: TRAIN CLASSIFIER
    print("\n Training SMOTE classifier...")

    smote_model = train_smote_classifier(
        X_resampled, y_resampled, CFG
    )

    # STEP 5: EVALUATE
    print("\n Evaluating SMOTE model...")

    evaluate_smote(
        smote_model,
        feature_extractor,
        test_loader,
        CFG,
        scaler=scaler
    )

    print("\n CONTRAST + HYBRID + SMOTE experiment completed successfully")
