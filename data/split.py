import pandas as pd
from sklearn.model_selection import train_test_split


# PRINT CLASS DISTRIBUTION

def print_distribution(df, name):
    print(f"\n{name} class distribution:")
    print(df["label"].value_counts(normalize=True).sort_index())
    print("*******************************")



# LESION-LEVEL SPLIT

def split_dataset(df, seed=42):
    """
    Splits dataset into train/val/test using lesion_id
    to avoid data leakage (same lesion in multiple sets)
    """


    # UNIQUE LESIONS

    lesion_df = df[["lesion_id", "dx"]].drop_duplicates()


    # TRAIN / TEMP (70 / 30)

    train_lesion, temp_lesion = train_test_split(
        lesion_df,
        test_size=0.30,
        stratify=lesion_df["dx"],
        random_state=seed
    )


    # VAL / TEST (15 / 15)

    val_lesion, test_lesion = train_test_split(
        temp_lesion,
        test_size=0.50,
        stratify=temp_lesion["dx"],
        random_state=seed
    )


    # MAP BACK TO FULL DATA

    train_df = df[df["lesion_id"].isin(train_lesion["lesion_id"])].reset_index(drop=True)
    val_df   = df[df["lesion_id"].isin(val_lesion["lesion_id"])].reset_index(drop=True)
    test_df  = df[df["lesion_id"].isin(test_lesion["lesion_id"])].reset_index(drop=True)


    # PRINT SPLIT INFO

    print("\n===== DATA SPLIT =====")
    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")


    # PRINT CLASS DISTRIBUTION

    print_distribution(train_df, "Train")
    print_distribution(val_df, "Validation")
    print_distribution(test_df, "Test")

    return train_df, val_df, test_df