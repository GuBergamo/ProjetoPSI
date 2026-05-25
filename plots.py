import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv("summary_results.csv")

# ============================================================
# AUC
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(df["Strategy"], df["AUC"])

plt.ylabel("AUC")

plt.title("AUC média por estratégia")

plt.tight_layout()

plt.show()

# ============================================================
# F1
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(df["Strategy"], df["F1"])

plt.ylabel("F1-score")

plt.title("F1-score médio por estratégia")

plt.tight_layout()

plt.show()

# ============================================================
# TEMPO
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(df["Strategy"], df["Training_Time"])

plt.ylabel("Tempo médio (s)")

plt.title("Tempo médio de treinamento")

plt.tight_layout()

plt.show()

# ============================================================
# TRADEOFF
# ============================================================

plt.figure(figsize=(7,6))

plt.scatter(
    df["Training_Time"],
    df["AUC"]
)

for i in range(len(df)):

    plt.text(
        df["Training_Time"][i],
        df["AUC"][i],
        df["Strategy"][i]
    )

plt.xlabel("Tempo médio (s)")

plt.ylabel("AUC")

plt.title("Tradeoff: performance vs custo computacional")

plt.tight_layout()

plt.show()