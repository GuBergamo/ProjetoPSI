import numpy as np
import pandas as pd
import librosa

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CSV_PATH = "balanced_dataset.csv"

SR = 16000

N_MFCC = 30

FRAME_LENGTH = int(0.025 * SR)   # 25 ms
HOP_LENGTH = int(0.005 * SR)     # 5 ms

# ============================================================
# CARREGAR DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Total de arquivos:", len(df))

# ============================================================
# LISTAS
# ============================================================

X_strategy_1 = []
X_strategy_2 = []
X_strategy_3 = []

y = []

# ============================================================
# CONTADORES
# ============================================================

processed = 0
failed = 0

# ============================================================
# LOOP
# ============================================================

for idx, row in df.iterrows():

    file_path = row["file"]
    label = row["label"]

    print(f"[{idx+1}/{len(df)}] {file_path}")

    try:

        # ====================================================
        # LOAD
        # ====================================================

        signal, sr = librosa.load(file_path, sr=SR)

        # ====================================================
        # VERIFICAÇÕES IMPORTANTES
        # ====================================================

        # áudio vazio
        if len(signal) == 0:
            print("Áudio vazio")
            failed += 1
            continue

        # NaN / inf
        if not np.isfinite(signal).all():
            print("Áudio contém NaN ou inf")
            failed += 1
            continue

        # silêncio total
        max_val = np.max(np.abs(signal))

        if max_val == 0:
            print("Áudio silencioso")
            failed += 1
            continue

        # ====================================================
        # NORMALIZAÇÃO
        # ====================================================

        signal = signal / max_val

        # ====================================================
        # MFCC
        # ====================================================

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        # ====================================================
        # DELTAS
        # ====================================================

        delta = librosa.feature.delta(mfcc)

        delta2 = librosa.feature.delta(mfcc, order=2)

        # ====================================================
        # ESTRATÉGIA 1
        # ====================================================

        strategy_1 = np.mean(mfcc, axis=1)

        # ====================================================
        # ESTRATÉGIA 2
        # ====================================================

        strategy_2 = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1)
        ])

        # ====================================================
        # ESTRATÉGIA 3
        # ====================================================

        strategy_3 = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ])

        # ====================================================
        # VERIFICAÇÃO FINAL
        # ====================================================

        if not np.isfinite(strategy_3).all():
            print("Features inválidas")
            failed += 1
            continue

        # ====================================================
        # SALVAR
        # ====================================================

        X_strategy_1.append(strategy_1)
        X_strategy_2.append(strategy_2)
        X_strategy_3.append(strategy_3)

        y.append(label)

        processed += 1

    except Exception as e:

        print("\nERRO:")
        print(file_path)
        print(e)
        print("\n")

        failed += 1

# ============================================================
# NUMPY
# ============================================================

X_strategy_1 = np.array(X_strategy_1)
X_strategy_2 = np.array(X_strategy_2)
X_strategy_3 = np.array(X_strategy_3)

y = np.array(y)

# ============================================================
# RESULTADOS
# ============================================================

print("\n===================================")
print("FINALIZADO")
print("===================================")

print("Processados:", processed)
print("Falhas:", failed)

print("\nShapes:")

print("Strategy 1:", X_strategy_1.shape)
print("Strategy 2:", X_strategy_2.shape)
print("Strategy 3:", X_strategy_3.shape)

print("Labels:", y.shape)

# ============================================================
# SALVAR
# ============================================================

np.save("X_strategy_1.npy", X_strategy_1)
np.save("X_strategy_2.npy", X_strategy_2)
np.save("X_strategy_3.npy", X_strategy_3)

np.save("y.npy", y)

print("\nFeatures salvas!")