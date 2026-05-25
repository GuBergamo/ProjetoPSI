import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv("systematic_results_parallel.csv")

# ============================================================
# 1. STRATEGY COMPARISON
# ============================================================

strategy_mean = df.groupby("Strategy")[
    "AUC_mean"
].mean()

plt.figure(figsize=(8,5))

plt.bar(
    strategy_mean.index,
    strategy_mean.values
)

plt.ylabel("Mean AUC")

plt.title("Strategy Comparison")

plt.tight_layout()

plt.show()

# ============================================================
# 2. GAMMA ANALYSIS
# ============================================================

plt.figure(figsize=(8,5))

for strategy in df["Strategy"].unique():

    sub = df[
        (df["Strategy"] == strategy) &
        (df["Kernel"] == "rbf") &
        (df["C"] == 1)
    ]

    plt.plot(
        sub["Gamma"],
        sub["AUC_mean"],
        marker='o',
        label=strategy
    )

plt.xscale("log")

plt.xlabel("Gamma")

plt.ylabel("AUC")

plt.title("Gamma Sensitivity")

plt.legend()

plt.tight_layout()

plt.show()

# ============================================================
# 3. C ANALYSIS
# ============================================================

plt.figure(figsize=(8,5))

for strategy in df["Strategy"].unique():

    sub = df[
        (df["Strategy"] == strategy) &
        (df["Kernel"] == "rbf") &
        (df["Gamma"] == 0.01)
    ]

    plt.plot(
        sub["C"],
        sub["AUC_mean"],
        marker='o',
        label=strategy
    )

plt.xscale("log")

plt.xlabel("C")

plt.ylabel("AUC")

plt.title("C Sensitivity")

plt.legend()

plt.tight_layout()

plt.show()

# ============================================================
# 4. TRAINING TIME
# ============================================================

time_mean = df.groupby("Strategy")[
    "Training_Time_mean"
].mean()

plt.figure(figsize=(8,5))

plt.bar(
    time_mean.index,
    time_mean.values
)

plt.ylabel("Training Time (s)")

plt.title("Average Training Time")

plt.tight_layout()

plt.show()

# ============================================================
# 5. KERNEL COMPARISON
# ============================================================

kernel_mean = df.groupby("Kernel")[
    "AUC_mean"
].mean()

plt.figure(figsize=(6,5))

plt.bar(
    kernel_mean.index,
    kernel_mean.values
)

plt.ylabel("Mean AUC")

plt.title("Kernel Comparison")

plt.tight_layout()

plt.show()

