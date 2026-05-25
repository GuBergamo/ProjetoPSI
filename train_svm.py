import numpy as np
import pandas as pd

from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

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
# CARREGAR FEATURES
# ============================================================

X1 = np.load("X_strategy_1.npy")
X2 = np.load("X_strategy_2.npy")
X3 = np.load("X_strategy_3.npy")

y = np.load("y.npy")

# ============================================================
# DICIONÁRIO DAS ESTRATÉGIAS
# ============================================================

strategies = {
    "Strategy_1": X1,
    "Strategy_2": X2,
    "Strategy_3": X3
}

# ============================================================
# HYPERPARAMETERS
# ============================================================

param_grid = {

    "svm__C": [0.1, 1, 10, 100],

    "svm__gamma": [0.001, 0.01, 0.1, 1],

    "svm__kernel": ["rbf", "linear"]

}

# ============================================================
# NESTED CV
# ============================================================

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ============================================================
# RESULTADOS
# ============================================================

results = []

# ============================================================
# LOOP DAS ESTRATÉGIAS
# ============================================================

for strategy_name, X in strategies.items():

    print("\n===================================")
    print(strategy_name)
    print("===================================")

    fold_number = 1

    # ========================================================
    # OUTER LOOP
    # ========================================================

    for train_idx, test_idx in outer_cv.split(X, y):

        print(f"\nFold externo {fold_number}")

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ====================================================
        # PIPELINE
        # ====================================================

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True))
        ])

        # ====================================================
        # GRID SEARCH
        # ====================================================

        grid_search = GridSearchCV(

            estimator=pipeline,

            param_grid=param_grid,

            cv=inner_cv,

            scoring="accuracy",

            n_jobs=-1
        )

        # ====================================================
        # TREINAMENTO
        # ====================================================

        grid_search.fit(X_train, y_train)

        # melhor modelo
        best_model = grid_search.best_estimator_

        print("Melhores parâmetros:")
        print(grid_search.best_params_)

        # ====================================================
        # PREDIÇÕES
        # ====================================================

        y_pred = best_model.predict(X_test)

        y_prob = best_model.predict_proba(X_test)[:, 1]

        # ====================================================
        # MATRIZ DE CONFUSÃO
        # ====================================================

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()

        # ====================================================
        # MÉTRICAS
        # ====================================================

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred)

        recall = recall_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)

        auc = roc_auc_score(y_test, y_prob)

        # ====================================================
        # PRINT
        # ====================================================

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        # ====================================================
        # SALVAR RESULTADOS
        # ====================================================

        results.append({

            "Strategy": strategy_name,

            "Fold": fold_number,

            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,

            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc,

            "Best_C":
                grid_search.best_params_["svm__C"],

            "Best_gamma":
                grid_search.best_params_["svm__gamma"],

            "Best_kernel":
                grid_search.best_params_["svm__kernel"]
        })

        fold_number += 1

# ============================================================
# DATAFRAME FINAL
# ============================================================

results_df = pd.DataFrame(results)

# ============================================================
# MÉDIAS
# ============================================================

summary_df = results_df.groupby("Strategy")[[
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC"
]].mean()

print("\n===================================")
print("MÉDIAS FINAIS")
print("===================================")

print(summary_df)

# ============================================================
# SALVAR CSV
# ============================================================

results_df.to_csv("all_results.csv", index=False)

summary_df.to_csv("summary_results.csv")

print("\nResultados salvos!")