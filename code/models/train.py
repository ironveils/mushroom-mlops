import time
from pathlib import Path

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Configuration
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "mushroom_catboost.cbm"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

TARGET = "class"

MLFLOW_EXPERIMENT = "mushroom-classification"

MODEL_PARAMS = {
    "iterations": 300,
    "depth": 8,
    "learning_rate": 0.1,
    "loss_function": "Logloss",
    "eval_metric": "MCC",
    "random_seed": RANDOM_STATE,
    "verbose": 50,
    "thread_count": -1,
    "allow_writing_files": False,
}


# Load data
print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)

print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)

print(f"Full train shape: {train_df.shape}")
print(f"Full test shape:  {test_df.shape}")


# Create training sample
if SAMPLE_SIZE < len(train_df):
    train_sample, _ = train_test_split(
        train_df,
        train_size=SAMPLE_SIZE,
        stratify=train_df[TARGET],
        random_state=RANDOM_STATE,
    )
else:
    train_sample = train_df

print(f"\nTraining sample shape: {train_sample.shape}")


# Prepare features
X_train = train_sample.drop(columns=[TARGET])
y_train = train_sample[TARGET].map({"e": 0, "p": 1})

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].map({"e": 0, "p": 1})


# Identify categorical features
categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print(f"\nNumber of features: {X_train.shape[1]}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Numeric features: {X_train.shape[1] - len(categorical_features)}")


# MLflow
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Train model
print("\nStarting model training...")

start_time = time.time()

with mlflow.start_run():
    mlflow.log_params(MODEL_PARAMS)
    mlflow.log_param("sample_size", len(train_sample))
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("categorical_features", len(categorical_features))

    model = CatBoostClassifier(**MODEL_PARAMS)

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
    )

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Evaluation
    print("\nEvaluating model on full test set...")

    y_pred = model.predict(X_test).flatten()

    mcc = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nTest Metrics: ")
    print(f"MCC:       {mcc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Train time: {training_time:.2f} sec")

    # Log metrics to MLflow
    mlflow.log_metric("mcc", mcc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("training_time_seconds", training_time)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Log model artifact to MLflow
    mlflow.log_artifact(MODEL_PATH)

    print("Model artifact logged to MLflow.")
    print("\nMLflow run completed.")