import pandas as pd

CSV_PATH = "Coswara-Data-master/combined_data.csv"

df = pd.read_csv(CSV_PATH)

# ver todas as classes únicas
print(df["covid_status"].unique())

print("\nContagem por classe:")
print(df["covid_status"].value_counts())

