import pandas as pd
import numpy as np

# ============================================================
# LOAD CSVs
# ============================================================

old_df = pd.read_csv(
    "systematic_results.csv"
)

new_df = pd.read_csv(
    "systematic_results_parallel.csv"
)

# ============================================================
# SORT
# ============================================================

sort_columns = [
    "Strategy",
    "C",
    "Gamma",
    "Kernel"
]

old_df = old_df.sort_values(
    by=sort_columns
).reset_index(drop=True)

new_df = new_df.sort_values(
    by=sort_columns
).reset_index(drop=True)

# ============================================================
# CHECK SHAPES
# ============================================================

print("\n==============================")
print("SHAPE CHECK")
print("==============================")

print("Old:", old_df.shape)
print("New:", new_df.shape)

# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_cols = [

    "Accuracy_mean",
    "Precision_mean",
    "Recall_mean",
    "F1_mean",
    "AUC_mean",

    "Training_Time_mean",

    "TP_mean",
    "TN_mean",
    "FP_mean",
    "FN_mean"
]

# ============================================================
# COMPARE
# ============================================================

print("\n==============================")
print("COLUMN COMPARISON")
print("==============================")

all_equal = True

for col in numeric_cols:

    equal = np.allclose(
        old_df[col],
        new_df[col],
        atol=1e-10
    )

    print(f"{col}: {equal}")

    if not equal:

        all_equal = False

        diff = np.abs(
            old_df[col] - new_df[col]
        )

        print("Max difference:")
        print(diff.max())

# ============================================================
# FINAL RESULT
# ============================================================

print("\n==============================")

if all_equal:

    print("ALL RESULTS MATCH")

else:

    print("DIFFERENCES FOUND")

print("==============================")