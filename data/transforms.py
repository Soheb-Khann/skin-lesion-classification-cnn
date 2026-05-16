import albumentations as A
from albumentations.pytorch import ToTensorV2


# TRAIN TRANSFORMS
def get_train_transforms(cfg):

    if cfg.use_augmentation:

        return A.Compose([

            A.Resize(cfg.img_size, cfg.img_size),

            # Basic augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            # Optional crop (keep if you used earlier)
            A.RandomCrop(cfg.img_size, cfg.img_size, p=1.0),

            # Strong rotation
            A.Rotate(limit=30, p=0.7),

            # Geometric
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.1, 0.1),
                rotate=(-30, 30),
                p=0.5
            ),

            # Color
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.5
            ),



            A.Normalize(),
            ToTensorV2()
        ])

    else:
        return A.Compose([
            A.Resize(cfg.img_size, cfg.img_size),
            A.Normalize(),
            ToTensorV2()
        ])


# VALIDATION / TEST TRANSFORMS
def get_valid_transforms(cfg):
    return A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(),
        ToTensorV2()
    ])