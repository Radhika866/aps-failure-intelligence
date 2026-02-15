"""
APS Failure Prediction - ONE CLICK FINAL SCRIPT
Run once -> trains all models -> saves results
"""

# =====================================================
# IMPORTS
# =====================================================
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

warnings.filterwarnings("ignore", message=".*use_label_encoder.*")

# =====================================================
# CONFIG
# =====================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

MODEL_DIR = "model"
TRAIN_PATH = "models/data/aps_failure_training_set.csv"
TEST_PATH = "models/data/aps_failure_test_set.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# DATA LOADING
# =====================================================
def load_aps_file(path):
    """
    Automatically finds where APS data table starts.
    Works for all APS dataset versions.
    """

    # find correct header line
    with open(path, "r") as f:
        lines = f.readlines()

    header_line = None
    for i, line in enumerate(lines):
        if line.startswith("class,"):
            header_line = i
            break

    if header_line is None:
        raise ValueError("Could not find 'class' header in file.")

    # read starting from real header
    df = pd.read_csv(
        path,
        skiprows=header_line,
        na_values="na"
    )

    return df


def load_data(train_path, test_path):

    print("📂 Reading APS dataset...")

    train_df = load_aps_file(train_path)
    test_df = load_aps_file(test_path)

    print("Loaded columns:", train_df.columns[:5])

    y_train = train_df["class"].map({"neg": 0, "pos": 1})
    y_test = test_df["class"].map({"neg": 0, "pos": 1})

    X_train = train_df.drop("class", axis=1)
    X_test = test_df.drop("class", axis=1)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

    return X_train, X_test, y_train, y_test

# =====================================================
# METRICS
# =====================================================
def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    try:
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": auc,
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "MCC": matthews_corrcoef(y_test, preds)
    }


# =====================================================
# MAIN
# =====================================================
def main():

    print("🚀 APS FAILURE TRAINING STARTED")

    X_train, X_test, y_train, y_test = load_data(
        TRAIN_PATH,
        TEST_PATH
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    print("Class distribution:")
    print(y_train.value_counts())

    # save feature names (for Streamlit)
    joblib.dump(
        X_train.columns.tolist(),
        f"{MODEL_DIR}/feature_names.pkl"
    )

    # imbalance ratio
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # =================================================
    # MODELS
    # =================================================
    models = {

        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "knn": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=3,
                weights="distance"
            ))
        ]),

        "naive_bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),

        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ]),

        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=200,
                eval_metric="logloss",
                scale_pos_weight=pos_weight,
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])
    }

    # =================================================
    # TRAIN LOOP
    # =================================================
    results = []

    for name, model in models.items():

        print(f"\n🔹 Training {name} ...")
        start = time.time()

        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        metrics["Model"] = name
        metrics["Train_Time_sec"] = round(time.time() - start, 2)

        results.append(metrics)

        joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")

    # =================================================
    # SAVE RESULTS
    # =================================================
    results_df = pd.DataFrame(results)

    num_cols = results_df.select_dtypes(include=np.number).columns
    results_df[num_cols] = results_df[num_cols].round(4)

    results_df = (
        results_df
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )

    results_df.insert(0, "Rank", range(1, len(results_df)+1))

    results_df.to_csv(f"{MODEL_DIR}/model_results.csv", index=False)

    print("\n✅ TRAINING COMPLETE")
    print(results_df)


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()
