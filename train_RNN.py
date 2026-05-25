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


X = np.load("features/X_rnn.npy")
y = np.load("features/y.npy")

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
# SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame({

    "Model": ["LSTM"],

    "N_MFCC": [N_MFCC],

    "Hidden_Size": [HIDDEN_SIZE],

    "Num_Layers": [NUM_LAYERS],

    "Batch_Size": [BATCH_SIZE],

    "Epochs": [EPOCHS],

    "Learning_Rate": [LEARNING_RATE],

    "TP": [tp],

    "TN": [tn],

    "FP": [fp],

    "FN": [fn],

    "Accuracy": [accuracy],

    "Precision": [precision],

    "Recall": [recall],

    "F1_Score": [f1],

    "AUC_ROC": [auc]
})


results_df.to_csv(
    "results/RNN_results.csv",
    index=False
)