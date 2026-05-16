import torch
import os


class CFG:


    # PATHS


    data_dir = r"C:\Users\Rajeshwari\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2"

    img_dirs = [
        os.path.join(data_dir, "HAM10000_images_part_1"),
        os.path.join(data_dir, "HAM10000_images_part_2")
    ]

    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")



    # TRAINING PARAMETERS


    img_size = 224
    batch_size = 32
    epochs = 25
    lr = 1e-3
    num_classes = 7

    num_tabular_features = 9



    # EXPERIMENT TYPE


    experiment = "baseline"



    # PIPELINE FLAGS


    use_metadata = False
    use_handcrafted = False
    use_hair_removal = False

    use_augmentation = False
    use_smote = False
    use_weighted_loss = False
    use_contrast = False



    # DEVICE


    device = "cuda" if torch.cuda.is_available() else "cpu"




# CONFIG UPDATER


def update_cfg(cfg):
    aliases = {
        "hybrid_weighted": "weighted_hybrid",
        "contrast_weighted": "contrast_hybrid_weighted",
        "contrast_smote": "contrast_hybrid_smote",
    }

    if cfg.experiment in aliases:
        canonical_name = aliases[cfg.experiment]
        print(f" Remapping experiment '{cfg.experiment}' -> '{canonical_name}'")
        cfg.experiment = canonical_name

    # RESET EVERYTHING
    cfg.use_metadata = False
    cfg.use_handcrafted = False
    cfg.use_hair_removal = False

    cfg.use_augmentation = False
    cfg.use_smote = False
    cfg.use_weighted_loss = False
    cfg.use_contrast = False



    # DATASET SWITCH


    if "contrast" in cfg.experiment:
        cfg.use_contrast = True

        cfg.img_dirs = [
            os.path.join(cfg.data_dir, "CLAHE_images_with_hair")
        ]

        print(" Using CONTRAST dataset (CLAHE)")
    else:
        cfg.img_dirs = [
            os.path.join(cfg.data_dir, "HAM10000_images_part_1"),
            os.path.join(cfg.data_dir, "HAM10000_images_part_2")
        ]



    # BASELINE


    if cfg.experiment in ["baseline", "contrast_baseline"]:
        cfg.num_tabular_features = 1



    # HYBRID


    elif cfg.experiment in ["hybrid", "contrast_hybrid"]:
        cfg.use_metadata = True
        cfg.use_handcrafted = True
        cfg.use_hair_removal = True
        cfg.num_tabular_features = 9



    # WEIGHTED LOSS


    elif cfg.experiment in ["weighted_hybrid", "contrast_hybrid_weighted"]:
        cfg.use_metadata = True
        cfg.use_handcrafted = True
        cfg.use_hair_removal = True
        cfg.use_weighted_loss = True
        cfg.num_tabular_features = 9

    # WEIGHTED BASIC (IMAGE ONLY)
    elif cfg.experiment == "weighted":
        cfg.use_metadata = False
        cfg.use_handcrafted = False
        cfg.use_hair_removal = False
        cfg.use_augmentation = False
        cfg.use_smote = False
        cfg.use_weighted_loss = True
        cfg.num_tabular_features = 1

    # AUGMENTATION (IMAGE ONLY)
    elif cfg.experiment == "augmentation":
        cfg.use_metadata = False
        cfg.use_handcrafted = False
        cfg.use_hair_removal = False
        cfg.use_augmentation = True
        cfg.num_tabular_features = 1

    # AUGMENTATION
    elif cfg.experiment in ["hybrid_augmentation", "contrast_hybrid_augmentation"]:
        cfg.use_metadata = True
        cfg.use_handcrafted = True
        cfg.use_hair_removal = True
        cfg.use_augmentation = True
        cfg.num_tabular_features = 9



    # SMOTE


    elif cfg.experiment in ["smote", "contrast_hybrid_smote"]:
        cfg.use_metadata = True
        cfg.use_handcrafted = True
        cfg.use_hair_removal = True
        cfg.use_smote = True
        cfg.num_tabular_features = 9



    # SAFETY


    else:
        raise ValueError(f"Unknown experiment type: {cfg.experiment}")

    return cfg




# PRINT PIPELINE


def print_pipeline(cfg):
    print("\n***** PIPELINE CONFIG *****")
    print(f"Experiment: {cfg.experiment}")

    print(f"Using Contrast Dataset: {cfg.use_contrast}")
    print(f"Using Metadata: {cfg.use_metadata}")
    print(f"Using Handcrafted Features: {cfg.use_handcrafted}")
    print(f"Hair Removal: {cfg.use_hair_removal}")

    print(f"Augmentation: {cfg.use_augmentation}")
    print(f"SMOTE: {cfg.use_smote}")
    print(f"Weighted Loss: {cfg.use_weighted_loss}")

    print(f"Device: {cfg.device}")
    print("*******************************\n")
