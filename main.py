if __name__ == "__main__":
    from configs.config import CFG, update_cfg, print_pipeline

    from data.dataset import prepare_dataframe, SkinDataset
    from data.split import split_dataset

    from training.dataloader import get_dataloaders

    from models.hybrid_model import HybridModel

    import torch
    from training.smote_pipeline import extract_features
    from training.smote_pipeline import (
        extract_features,
        apply_smote,
        train_smote_classifier,
        evaluate_smote
    )

    CFG.experiment = "baseline"
    CFG = update_cfg(CFG)
    print_pipeline(CFG)

    df, label_map = prepare_dataframe(CFG)

    print("\n[DATAFRAME CHECK]")
    print(df.head())

    dataset = SkinDataset(df, CFG)

    img, feats, label = dataset[0]

    print("\n[DATASET CHECK]")
    print("Image shape:", img.shape)
    print("Feature shape:", feats.shape)
    print("Label:", label)


if __name__ == "__main__":
    from configs.config import CFG
    from data.dataset import prepare_dataframe

    df, _ = prepare_dataframe(CFG)

    train_df, val_df, test_df = split_dataset(df)

    print("\n[SPLIT CHECK]")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    print("\nTrain class distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation class distribution:")
    print(val_df["label"].value_counts())

    print("\nTest class distribution:")
    print(test_df["label"].value_counts())


if __name__ == "__main__":
    from configs.config import CFG, update_cfg
    from data.dataset import prepare_dataframe
    from data.split import split_dataset

    CFG = update_cfg(CFG)

    df, _ = prepare_dataframe(CFG)
    train_df, val_df, test_df = split_dataset(df)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, CFG
    )

    imgs, feats, labels = next(iter(train_loader))

    print("\n[DATALOADER CHECK]")
    print("Images:", imgs.shape)
    print("Features:", feats.shape)
    print("Labels:", labels.shape)

if __name__ == "__main__":
    import torch
    from configs.config import CFG, update_cfg

    CFG = update_cfg(CFG)

    model = HybridModel(CFG)

    img = torch.randn(2, 3, 224, 224)
    feats = torch.randn(2, CFG.num_tabular_features)

    out = model(img, feats)

    print("\n[FEATURE VERIFICATION]")

    print("Total feature length:", feats.shape[0])
    print("Expected feature length:", CFG.num_tabular_features)

    # Optional deeper debug
    print("Feature vector:", feats)

    print("\n[MODEL CHECK]")
    print("Input image:", img.shape)
    print("Input features:", feats.shape)
    print("Output:", out.shape)

if __name__ == "__main__":
    from configs.config import CFG, update_cfg
    from data.dataset import prepare_dataframe
    from data.split import split_dataset
    from training.dataloader import get_dataloaders
    from models.hybrid_model import HybridModel

    CFG.experiment = "smote"
    CFG = update_cfg(CFG)

    df, _ = prepare_dataframe(CFG)
    train_df, val_df, test_df = split_dataset(df)

    train_loader, _, _ = get_dataloaders(train_df, val_df, test_df, CFG)

    model = HybridModel(CFG).to(CFG.device)

    X, y = extract_features(model, train_loader, CFG.device)

    print("\n[SMOTE FEATURE CHECK]")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)

    X_res, y_res, _ = apply_smote(X, y)

    print("\n[SMOTE CHECK]")
    print("After SMOTE:", X_res.shape, y_res.shape)

if __name__ == "__main__":
    from configs.config import CFG, update_cfg
    from data.dataset import prepare_dataframe
    from data.split import split_dataset
    from training.dataloader import get_dataloaders
    from models.hybrid_model import HybridModel

    CFG.experiment = "smote"
    CFG = update_cfg(CFG)

    df, _ = prepare_dataframe(CFG)
    train_df, val_df, test_df = split_dataset(df)

    train_loader, _, _ = get_dataloaders(train_df, val_df, test_df, CFG)

    model = HybridModel(CFG).to(CFG.device)

    X, y = extract_features(model, train_loader, CFG.device)

    print("\n[SMOTE FEATURE CHECK]")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)

    X_res, y_res, _ = apply_smote(X, y)

    print("\n[SMOTE CHECK]")
    print("After SMOTE:", X_res.shape, y_res.shape)
