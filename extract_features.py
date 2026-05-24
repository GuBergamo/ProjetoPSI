import numpy as np
import pandas as pd
import librosa

# carregar CSV balanceado
df = pd.read_csv("balanced_dataset.csv")

# listas finais
X = []
y = []

# parâmetros
SR = 16000
N_MFCC = 13

for idx, row in df.iterrows():

    file_path = row["file"]
    label = row["label"]

    try:
        # carregar áudio
        signal, sr = librosa.load(file_path, sr=SR)

        # =========================
        # MFCC
        # =========================
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # =========================
        # DELTA
        # =========================
        delta = librosa.feature.delta(mfcc)

        # =========================
        # DELTA-DELTA
        # =========================
        delta2 = librosa.feature.delta(mfcc, order=2)

        # =========================
        # AGREGAÇÃO TEMPORAL
        # média no tempo
        # =========================
        mfcc_mean = np.mean(mfcc, axis=1)

        # opção completa
        features = np.concatenate([
            mfcc_mean,
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ])

        X.append(features)
        y.append(label)

    except Exception as e:
        print("Erro em:", file_path)
        print(e)

# transformar em numpy
X = np.array(X)
y = np.array(y)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# salvar
np.save("X.npy", X)
np.save("y.npy", y)

print("Features salvas!")