import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CREATE FOLDER
# ============================================================

os.makedirs(
    "results/plots",
    exist_ok=True
)

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(
    "results/systematic_results_parallel.csv"
)

# ============================================================
# BEST RESULTS
# ============================================================

best_df = df.sort_values(

    by="AUC_mean",

    ascending=False
)

print("\nTop 10 Results:\n")

print(
    best_df[
        [
            "Strategy",
            "C",
            "Gamma",
            "Kernel",
            "AUC_mean"
        ]
    ].head(10)
)

# ============================================================
# STRATEGY COMPARISON
# ============================================================

strategy_auc = df.groupby(

    "Strategy"

)["AUC_mean"].mean()

plt.figure(figsize=(8,5))

strategy_auc.plot(
    kind="bar"
)

plt.ylabel("Mean AUC")

plt.title("Strategy Comparison")

plt.tight_layout()

plt.savefig(
    "results/plots/strategy_comparison.png",
    dpi=300
)

plt.close()

# ============================================================
# GAMMA ANALYSIS
# ============================================================

for strategy in df["Strategy"].unique():

    subset = df[

        df["Strategy"] == strategy
    ]

    gamma_auc = subset.groupby(

        "Gamma"

    )["AUC_mean"].mean()

    plt.figure(figsize=(8,5))

    gamma_auc.plot(
        marker="o"
    )

    plt.ylabel("Mean AUC")

    plt.title(
        f"Gamma Analysis - {strategy}"
    )

    plt.tight_layout()

    plt.savefig(

        f"results/plots/gamma_{strategy}.png",

        dpi=300
    )

    plt.close()

# ============================================================
# C ANALYSIS
# ============================================================

for strategy in df["Strategy"].unique():

    subset = df[

        df["Strategy"] == strategy
    ]

    C_auc = subset.groupby(

        "C"

    )["AUC_mean"].mean()

    plt.figure(figsize=(8,5))

    C_auc.plot(
        marker="o"
    )

    plt.ylabel("Mean AUC")

    plt.title(
        f"C Analysis - {strategy}"
    )

    plt.tight_layout()

    plt.savefig(

        f"results/plots/C_{strategy}.png",

        dpi=300
    )

    plt.close()

# ============================================================
# KERNEL ANALYSIS
# ============================================================

for strategy in df["Strategy"].unique():

    subset = df[

        df["Strategy"] == strategy
    ]

    kernel_auc = subset.groupby(

        "Kernel"

    )["AUC_mean"].mean()

    plt.figure(figsize=(8,5))

    kernel_auc.plot(
        kind="bar"
    )

    plt.ylabel("Mean AUC")

    plt.title(
        f"Kernel Analysis - {strategy}"
    )

    plt.tight_layout()

    plt.savefig(

        f"results/plots/kernel_{strategy}.png",

        dpi=300
    )

    plt.close()

# ============================================================
# TRAINING TIME
# ============================================================

time_df = df.groupby(

    "Strategy"

)["Training_Time_mean"].mean()

plt.figure(figsize=(8,5))

time_df.plot(
    kind="bar"
)

plt.ylabel("Time (s)")

plt.title("Training Time Comparison")

plt.tight_layout()

plt.savefig(
    "results/plots/training_time.png",
    dpi=300
)

plt.close()

# ============================================================
# FINISH
# ============================================================

print("\nPlots saved in:")

print("results/plots/")