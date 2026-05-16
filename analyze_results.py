import pandas as pd
import os



# PATH SETUP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(BASE_DIR, "results", "results.csv")


# LOAD RESULTS

if not os.path.exists(results_path):
    print(" results.csv not found!")
    exit()

df = pd.read_csv(results_path)

print("\n Loaded results.csv")


# FILTER TEST RESULTS

df_test = df[df["stage"] == "test"].copy()

if df_test.empty:
    print(" No test results found!")
    exit()


# CLEAN DATA

df_test["test_accuracy"] = pd.to_numeric(df_test["test_accuracy"], errors="coerce")
df_test["test_f1"] = pd.to_numeric(df_test["test_f1"], errors="coerce")
df_test["test_recall"] = pd.to_numeric(df_test["test_recall"], errors="coerce")

df_test = df_test.dropna(subset=["test_f1"])


# ADD TYPE (NORMAL vs CONTRAST)

def get_type(exp):
    if "contrast" in exp:
        return "contrast"
    return "normal"

df_test["type"] = df_test["experiment"].apply(get_type)


# SORT BY BEST MODEL

df_sorted = df_test.sort_values(by="test_f1", ascending=False)


# FINAL TABLE (ALL MODELS)

final_df = df_sorted[
    ["experiment", "type", "test_accuracy", "test_f1", "test_recall"]
].reset_index(drop=True)

print("\n===== FINAL MODEL COMPARISON (ALL) =====\n")
print(final_df.to_string(index=False))


# BEST MODEL

best_model = final_df.iloc[0]

print("\nBEST MODEL:")
print(best_model)


# GROUPED COMPARISON

print("\n===== GROUPED COMPARISON =====\n")

groups = {
    "baseline": ["baseline", "contrast_baseline"],
    "hybrid": ["hybrid", "contrast_hybrid"],
    "augmentation": ["hybrid_augmentation", "contrast_hybrid_augmentation"],
    "weighted": ["weighted_hybrid", "contrast_hybrid_weighted"],
    "smote": ["smote", "contrast_hybrid_smote"]
}

for group_name, exps in groups.items():
    subset = final_df[final_df["experiment"].isin(exps)]

    if not subset.empty:
        print(f"\n--- {group_name.upper()} ---")
        print(subset.to_string(index=False))


# SAVE RESULTS

output_path = os.path.join(BASE_DIR, "results", "final_comparison.csv")
final_df.to_csv(output_path, index=False)

print("\n Final comparison saved at:", output_path)
