import numpy as np
import librosa
import torch
import torch.nn as nn
import time
import pandas as pd

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# DEVICE / GPU INFO
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n==============================")
print("DEVICE INFO")
print("==============================")

print("Torch version:")
print(torch.__version__)

print("\nSelected device:")
print(DEVICE)

if torch.cuda.is_available():

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    print("\nCUDA version:")
    print(torch.version.cuda)

else:

    print("\nCUDA not available")

# ============================================================
# PARAMETERS
# ============================================================

SR = 16000

N_MFCC = 30

FRAME_LENGTH = int(0.025 * SR)

HOP_LENGTH = int(0.005 * SR)

MAX_LEN = 1200

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 0.001

HIDDEN_SIZE = 64

NUM_LAYERS = 1

# ============================================================
# LOAD FILES
# ============================================================

df = pd.read_csv("balanced_dataset.csv")

files = df["file"].values

labels = df["label"].values

print("\nTotal files:", len(files))

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_mfcc_sequence(file_path):

    try:

        signal, sr = librosa.load(
            file_path,
            sr=SR
        )

        # ================================================
        # REMOVE INVALID SIGNALS
        # ================================================

        if len(signal) < 1000:
            return None

        if not np.isfinite(signal).all():
            return None

        # ================================================
        # NORMALIZATION
        # ================================================

        max_value = np.max(np.abs(signal))

        if max_value > 0:
            signal = signal / max_value

        # ================================================
        # MFCC
        # ================================================

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        # shape:
        # (N_MFCC, time_frames)

        mfcc = mfcc.T

        # ================================================
        # REMOVE NaNs
        # ================================================

        if np.isnan(mfcc).any():
            return None

        if np.isinf(mfcc).any():
            return None

        # ================================================
        # FIXED LENGTH
        # ================================================

        if mfcc.shape[0] > MAX_LEN:

            mfcc = mfcc[:MAX_LEN]

        else:

            pad_size = MAX_LEN - mfcc.shape[0]

            padding = np.zeros(
                (pad_size, N_MFCC)
            )

            mfcc = np.vstack([
                mfcc,
                padding
            ])

        return mfcc.astype(np.float32)

    except Exception as e:

        print("\nErro em:", file_path)
        print(e)

        return None

# ============================================================
# BUILD DATASET
# ============================================================

X = []
y = []

print("\n==============================")
print("EXTRACTING FEATURES")
print("==============================")

for i, (file_path, label) in enumerate(zip(files, labels)):

    mfcc_seq = extract_mfcc_sequence(file_path)

    if mfcc_seq is not None:

        X.append(mfcc_seq)

        y.append(label)

    if (i + 1) % 100 == 0:

        print(f"{i+1}/{len(files)} processed")

X = np.array(X)

y = np.array(y)

print("\nFinal dataset shape:")
print(X.shape)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42
)

print("\nTrain samples:", len(X_train))
print("Test samples :", len(X_test))

# ============================================================
# PYTORCH DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X)

        self.y = torch.tensor(y).float()

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

# ============================================================
# DATALOADERS
# ============================================================

train_dataset = AudioDataset(
    X_train,
    y_train
)

test_dataset = AudioDataset(
    X_test,
    y_test
)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False
)

# ============================================================
# LSTM MODEL
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=N_MFCC,

            hidden_size=HIDDEN_SIZE,

            num_layers=NUM_LAYERS,

            batch_first=True
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out, (hidden, cell) = self.lstm(x)

        # hidden shape:
        # (num_layers, batch, hidden_size)

        hidden = hidden[-1]

        out = self.fc(hidden)

        out = self.sigmoid(out)

        return out.squeeze()

# ============================================================
# MODEL
# ============================================================

model = LSTMClassifier().to(DEVICE)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)

# ============================================================
# TRAINING
# ============================================================

print("\n==============================")
print("TRAINING")
print("==============================")

for epoch in range(EPOCHS):

    epoch_start = time.time()

    model.train()

    running_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)

        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_end = time.time()

    epoch_time = epoch_end - epoch_start

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )

# ============================================================
# EVALUATION
# ============================================================

print("\n==============================")
print("EVALUATION")
print("==============================")

model.eval()

all_preds = []

all_probs = []

all_targets = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)

        outputs = model(X_batch)

        probs = outputs.cpu().numpy()

        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs)

        all_preds.extend(preds)

        all_targets.extend(y_batch.numpy())

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_targets,
    all_preds
)

precision = precision_score(
    all_targets,
    all_preds
)

recall = recall_score(
    all_targets,
    all_preds
)

f1 = f1_score(
    all_targets,
    all_preds
)

auc = roc_auc_score(
    all_targets,
    all_probs
)

tn, fp, fn, tp = confusion_matrix(
    all_targets,
    all_preds
).ravel()

# ============================================================
# RESULTS
# ============================================================

print("\n==============================")
print("LSTM RESULTS")
print("==============================")

print(f"TP : {tp}")
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")

print()

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")