import pandas as pd

DATA_PATH = "data/raw/data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(df["class"].value_counts())
print(df["class"].value_counts(normalize=True))

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)
print("Duplicate rows:", df.duplicated().sum())

categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
categorical_columns.remove("class")

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

for column in categorical_columns:
    print(f"\n--- {column} ---")
    print("Unique values:", df[column].nunique(dropna=False))
    print(df[column].value_counts(dropna=False).head(15))

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

for column in numerical_columns:
    print(f"\n--- {column} ---")
    print(df[column].describe())