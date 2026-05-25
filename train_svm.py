import numpy as np
import pandas as pd
import time

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# LOAD FEATURES
# ============================================================

X1 = np.load("X_strategy_1.npy")
X2 = np.load("X_strategy_2.npy")
X3 = np.load("X_strategy_3.npy")

y = np.load("y.npy")

# ============================================================
# STRATEGIES
# ============================================================

strategies = {
    "MFCC": X1,
    "MFCC_DELTA": X2,
    "MFCC_DELTA_DELTA": X3
}

# ============================================================
# HYPERPARAMETERS
# ============================================================

C_values = [0.1, 1, 10, 100]

gamma_values = [0.001, 0.01, 0.1, 1]

kernel_values = ["linear", "rbf"]

# ============================================================
# OUTER CV
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ============================================================
# RESULTS
# ============================================================

results = []

# ============================================================
# MAIN LOOP
# ============================================================

for strategy_name, X in strategies.items():

    print("\n========================================")
    print(strategy_name)
    print("========================================")

    for C in C_values:

        for gamma in gamma_values:

            for kernel in kernel_values:

                print("\n--------------------------------")
                print(f"C={C}")
                print(f"gamma={gamma}")
                print(f"kernel={kernel}")
                print("--------------------------------")

                fold_number = 1

                fold_accuracies = []
                fold_precisions = []
                fold_recalls = []
                fold_f1s = []
                fold_aucs = []

                fold_times = []

                fold_tps = []
                fold_tns = []
                fold_fps = []
                fold_fns = []

                # ============================================
                # OUTER CV
                # ============================================

                for train_idx, test_idx in outer_cv.split(X, y):

                    start_time = time.time()

                    X_train = X[train_idx]
                    X_test = X[test_idx]

                    y_train = y[train_idx]
                    y_test = y[test_idx]

                    # ========================================
                    # PIPELINE
                    # ========================================

                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),

                        ("svm", SVC(
                            C=C,
                            gamma=gamma,
                            kernel=kernel,
                            probability=True
                        ))
                    ])

                    # ========================================
                    # TRAIN
                    # ========================================

                    pipeline.fit(X_train, y_train)

                    # ========================================
                    # PREDICT
                    # ========================================

                    y_pred = pipeline.predict(X_test)

                    y_prob = pipeline.predict_proba(X_test)[:,1]

                    # ========================================
                    # CONFUSION MATRIX
                    # ========================================

                    tn, fp, fn, tp = confusion_matrix(
                        y_test,
                        y_pred
                    ).ravel()

                    # ========================================
                    # METRICS
                    # ========================================

                    accuracy = accuracy_score(
                        y_test,
                        y_pred
                    )

                    precision = precision_score(
                        y_test,
                        y_pred
                    )

                    recall = recall_score(
                        y_test,
                        y_pred
                    )

                    f1 = f1_score(
                        y_test,
                        y_pred
                    )

                    auc = roc_auc_score(
                        y_test,
                        y_prob
                    )

                    # ========================================
                    # TIME
                    # ========================================

                    end_time = time.time()

                    elapsed = end_time - start_time

                    # ========================================
                    # SAVE FOLD RESULTS
                    # ========================================

                    fold_accuracies.append(accuracy)
                    fold_precisions.append(precision)
                    fold_recalls.append(recall)
                    fold_f1s.append(f1)
                    fold_aucs.append(auc)

                    fold_times.append(elapsed)

                    fold_tps.append(tp)
                    fold_tns.append(tn)
                    fold_fps.append(fp)
                    fold_fns.append(fn)

                    print(
                        f"Fold {fold_number} | "
                        f"AUC={auc:.4f} | "
                        f"Acc={accuracy:.4f}"
                    )

                    fold_number += 1

                # ============================================
                # SAVE AVERAGE RESULTS
                # ============================================

                results.append({

                    "Strategy": strategy_name,

                    "C": C,

                    "Gamma": gamma,

                    "Kernel": kernel,

                    "Accuracy_mean":
                        np.mean(fold_accuracies),

                    "Precision_mean":
                        np.mean(fold_precisions),

                    "Recall_mean":
                        np.mean(fold_recalls),

                    "F1_mean":
                        np.mean(fold_f1s),

                    "AUC_mean":
                        np.mean(fold_aucs),

                    "Training_Time_mean":
                        np.mean(fold_times),

                    "TP_mean":
                        np.mean(fold_tps),

                    "TN_mean":
                        np.mean(fold_tns),

                    "FP_mean":
                        np.mean(fold_fps),

                    "FN_mean":
                        np.mean(fold_fns)
                })

# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    "systematic_results.csv",
    index=False
)

print("\n========================================")
print("RESULTADOS SALVOS")
print("========================================")