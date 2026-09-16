import pandas as pd

DATA_PATH = "data/raw/data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("Missing values: ")
print(df.isnull().sum())

print("Target distribution: ")
print(df["class"].value_counts())
print(df["class"].value_counts(normalize=True))

print("Duplicate rows:", df.duplicated().sum())

categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
categorical_columns.remove("class")

print("Categorical features: ")
for column in categorical_columns:
    print(f"\n--- {column} ---")
    print("Unique values:", df[column].nunique(dropna=False))
    print(df[column].value_counts(dropna=False).head(15))

print("Numerical features: ")
numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

for column in numerical_columns:
    print(f"\n--- {column} ---")
    print(df[column].describe())