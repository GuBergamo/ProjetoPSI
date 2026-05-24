import pandas as pd
import os

CSV_PATH = "Coswara-Data-master/combined_data.csv"
DATA_PATH = "Coswara-Data-master/Extracted_data"

N = 680

df = pd.read_csv(CSV_PATH)

# classes
positive_classes = [
    "positive_mild",
    "positive_moderate",
    "positive_asymp"
]

negative_classes = ["healthy"]

# filtrar
df_filtered = df[df["covid_status"].isin(positive_classes + negative_classes)]

df_pos = df_filtered[df_filtered["covid_status"].isin(positive_classes)]
df_neg = df_filtered[df_filtered["covid_status"].isin(negative_classes)]

print("Positivos disponíveis:", len(df_pos))
print("Negativos disponíveis:", len(df_neg))

# balancear
df_pos_sample = df_pos.sample(n=min(N, len(df_pos)), random_state=42)
df_neg_sample = df_neg.sample(n=min(N, len(df_neg)), random_state=42)

df_balanced = pd.concat([df_pos_sample, df_neg_sample])

print("Total balanceado:", len(df_balanced))


id_to_path = {}

for root, dirs, files in os.walk(DATA_PATH):
    folder_name = os.path.basename(root)
    id_to_path[folder_name] = root


def find_audio_files(df):
    file_paths = []
    labels = []

    for _, row in df.iterrows():
        user_id = row["id"]
        status = row["covid_status"]

        if user_id not in id_to_path:
            continue

        folder = id_to_path[user_id]

        for file in os.listdir(folder):
            if file == "counting-normal.wav":
                file_paths.append(os.path.join(folder, file))

                # label correto
                label = 1 if status in positive_classes else 0
                labels.append(label)

                break  # pega só 1 arquivo por usuário

    return file_paths, labels


files, labels = find_audio_files(df_balanced)

print("Arquivos encontrados:", len(files))

dataset_df = pd.DataFrame({
    "file": files,
    "label": labels
})

dataset_df.to_csv("balanced_dataset.csv", index=False)

print("Dataset salvo!")
