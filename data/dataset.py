import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# IMAGE PATH MAPPING
def build_image_path_dict(img_dirs):
    image_paths = {}

    for folder in img_dirs:
        for img_name in os.listdir(folder):
            image_id = img_name.split(".")[0]
            image_paths[image_id] = os.path.join(folder, img_name)

    return image_paths


# LOAD + MERGE DATA
def load_dataframe(cfg):
    df = pd.read_csv(cfg.metadata_path)

    image_dict = build_image_path_dict(cfg.img_dirs)

    df["image_path"] = df["image_id"].map(image_dict)

    return df


# LABEL ENCODING
def encode_labels(df):
    label_map = {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6
    }

    df["label"] = df["dx"].map(label_map)

    return df, label_map


# FINAL DATA PREP
def prepare_dataframe(cfg):
    df = load_dataframe(cfg)

    missing = df["image_path"].isnull().sum()
    print(f"Missing images: {missing}")

    df["age"] = df["age"].fillna(df["age"].median())
    df["sex"] = df["sex"].fillna("unknown")

    df, label_map = encode_labels(df)

    return df, label_map


# HAIR REMOVAL
def hair_removal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    result = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

    return result


# HANDCRAFTED FEATURES
def extract_color_features(image):
    image = image / 255.0

    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])

    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv / 255.0

    mean_hue = np.mean(hsv[:, :, 0])
    hue_var = np.var(hsv[:, :, 0])

    mean_sat = np.mean(hsv[:, :, 1])
    sat_var = np.var(hsv[:, :, 1])

    return np.array([
        mean_r, mean_g, mean_b,
        mean_hue, hue_var,
        mean_sat, sat_var
    ], dtype=np.float32)


# METADATA PROCESSING
def process_metadata(row):
    age = row["age"] / 100.0
    sex = 1 if row["sex"] == "male" else 0

    return np.array([age, sex], dtype=np.float32)


def tensor_to_uint8_image(img_tensor):
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img_np = ((img_np * IMAGENET_STD) + IMAGENET_MEAN) * 255.0
    return np.clip(img_np, 0, 255).astype(np.uint8)


# DATASET CLASS
class SkinDataset(Dataset):
    def __init__(self, df, cfg, transform=None, split="train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.split = split

        self.debug_done = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # LOAD IMAGE
        img = cv2.imread(row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # FLAGS
        aug_flag = False
        hair_flag = False
        metadata_flag = False
        handcrafted_flag = False

        # HAIR REMOVAL
        if self.cfg.use_hair_removal:
            img = hair_removal(img)
            hair_flag = True


        # APPLY TRANSFORMS FIRST

        if self.transform:
            img = self.transform(image=img)["image"]
            img_np = tensor_to_uint8_image(img)

            if self.split == "train" and self.cfg.use_augmentation:
                aug_flag = True
        else:
            img_np = img.copy()
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0


        # FEATURE BUILDING (AFTER AUGMENTATION)

        feature_list = []

        if self.cfg.use_handcrafted:
            feature_list.append(extract_color_features(img_np))
            handcrafted_flag = True

        if self.cfg.use_metadata:
            feature_list.append(process_metadata(row))
            metadata_flag = True

        if len(feature_list) > 0:
            features = np.concatenate(feature_list).astype(np.float32)
        else:
            features = np.zeros(self.cfg.num_tabular_features, dtype=np.float32)

        # CONVERT FEATURES
        features = torch.tensor(features, dtype=torch.float32).view(-1)

        # DEBUG PRINT (ONLY ONCE PER SPLIT)
        if not self.debug_done:

            print(f"\n[DATA CHECK - {self.split}]")
            print(" Image shape:", img.shape)
            print(" Feature shape:", features.shape)

            print(f" Hair removal: {hair_flag}")
            print(f" Augmentation: {aug_flag}")
            print(f" Metadata: {metadata_flag}")
            print(f" Handcrafted features: {handcrafted_flag}")

            print("*******************************\n")

            self.debug_done = True

        label = torch.tensor(row["label"], dtype=torch.long)

        return img, features, label
