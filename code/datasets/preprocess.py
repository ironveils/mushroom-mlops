from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
RAW_DATA_PATH = Path("data/raw/data.csv")
PROCESSED_DIR = Path("data/processed")

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Load data
print("Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Original shape: {df.shape}")

# Remove duplicates
duplicates = df.duplicated().sum()

if duplicates > 0:
    print(f"Removing {duplicates} duplicate rows...")
    df = df.drop_duplicates()
else:
    print("No duplicate rows found.")

# Separate target
TARGET = "class"
ID_COLUMN = "id"

y = df[TARGET]
X = df.drop(columns=[TARGET, ID_COLUMN])

# Handle missing values
print("\nHandling missing values...")

categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(include=["number"]).columns

for column in categorical_columns:
    X[column] = X[column].fillna("Unknown")

for column in numerical_columns:
    X[column] = X[column].fillna(X[column].median())

# Remove numerical outliers
print("\nRemoving numerical outliers...")

outlier_mask = pd.Series(True, index=X.index)

for column in numerical_columns:
    q1 = X[column].quantile(0.25)
    q3 = X[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    column_mask = (
        (X[column] >= lower_bound)
        & (X[column] <= upper_bound)
    )

    removed = (~column_mask).sum()

    print(
        f"{column}: "
        f"lower={lower_bound:.3f}, "
        f"upper={upper_bound:.3f}, "
        f"outliers={removed}"
    )

    outlier_mask &= column_mask

X = X[outlier_mask]
y = y.loc[X.index]

print(f"Shape after outlier removal: {X.shape}")

# Train / test split
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Restore target column
train = X_train.copy()
train[TARGET] = y_train

test = X_test.copy()
test[TARGET] = y_test

# Save processed data
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

# Summary
print("\nProcessing completed.")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

print("\nTrain target distribution:")
print(train[TARGET].value_counts(normalize=True))

print("\nTest target distribution:")
print(test[TARGET].value_counts(normalize=True))

print(f"\nSaved:")
print(f"  {TRAIN_PATH}")
print(f"  {TEST_PATH}")