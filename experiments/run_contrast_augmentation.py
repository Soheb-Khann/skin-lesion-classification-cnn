import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CFG, update_cfg, print_pipeline
from data.dataset import prepare_dataframe
from data.split import split_dataset
from training.dataloader import get_dataloaders
from models.hybrid_model import HybridModel
from training.trainer import train_model, evaluate_on_test
from utils.seed import set_seed

import torch


if __name__ == "__main__":

    # SEED
    set_seed(42)

    print("\n Running CONTRAST + HYBRID + AUGMENTATION experiment")

    # CONFIG
    CFG.experiment = "contrast_hybrid_augmentation"
    CFG = update_cfg(CFG)

    print_pipeline(CFG)

    # DATA
    df, _ = prepare_dataframe(CFG)

    train_df, val_df, test_df = split_dataset(df)

    # DATALOADERS
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, CFG
    )

    # MODEL
    model = HybridModel(CFG).to(CFG.device)

    # TRAIN
    train_model(model, train_loader, val_loader, train_df, CFG)

    # LOAD BEST MODEL
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "results", f"{CFG.experiment}_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=CFG.device))
        print("\n Loaded best model for evaluation")

    # TEST
    evaluate_on_test(model, test_loader, CFG.device, CFG)

    print("\n CONTRAST + HYBRID + AUGMENTATION experiment completed successfully")