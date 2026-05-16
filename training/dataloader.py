import torch
from torch.utils.data import DataLoader

from data.dataset import SkinDataset
from data.transforms import get_train_transforms, get_valid_transforms
from utils.seed import seed_worker


# CREATE DATALOADERS
def get_dataloaders(train_df, val_df, test_df, cfg):


    # TRANSFORM CONTROL

    if cfg.experiment in ["augmentation", "hybrid_augmentation", "contrast_hybrid_augmentation"]:
        train_transform = get_train_transforms(cfg)
        print(" Using AUGMENTATION transforms")
    else:
        train_transform = get_valid_transforms(cfg)
        print(" Using BASIC transforms (no augmentation)")


    # DATASETS

    train_dataset = SkinDataset(
        train_df,
        cfg,
        transform=train_transform,
        split="train"
    )

    val_dataset = SkinDataset(
        val_df,
        cfg,
        transform=get_valid_transforms(cfg),
        split="val"
    )

    test_dataset = SkinDataset(
        test_df,
        cfg,
        transform=get_valid_transforms(cfg),
        split="test"
    )


    # SEED GENERATOR

    g = torch.Generator()
    g.manual_seed(42)


    # DATALOADERS

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )


    # DEBUG INFO

    print("\n***** DATALOADER CHECK *****")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader