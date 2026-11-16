
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = "./data/zomato_dataset.csv"
OUTPUT_DIR = Path("./exploration/zomato_exploration")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n========== DATASET OVERVIEW ==========")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])
print("Dataset Size:", round(df.memory_usage(deep=True).sum() / 1024**2, 2), "MB")

# ==========================================
# 2. FIRST LOOK AT DATA
# ==========================================

print("\n========== FIRST 10 ROWS ==========")
print(df.head(10).to_string())

print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(column)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# ==========================================
# 3. COLUMN INFORMATION
# ==========================================

print("\n========== COLUMN DETAILS ==========")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Non-Null Count": df.count().values,
    "Missing Count": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

print(column_info.to_string(index=False))
column_info.to_csv(OUTPUT_DIR / "column_information.csv", index=False)

# ==========================================
# 4. UNIQUE VALUES
# ==========================================

print("\n========== UNIQUE VALUES ==========")

for column in df.columns:
    print(f"\n--- {column} ---")
    print("Unique count:", df[column].nunique(dropna=True))
    print("Sample values:")
    print(df[column].dropna().unique()[:30])

# ==========================================
# 5. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df) * 100).round(2)
})

print(missing.to_string())
missing.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)

# ==========================================
# 6. DUPLICATE RECORDS
# ==========================================

print("\n========== DUPLICATES ==========")

print("Duplicate Rows:", df.duplicated().sum())

# ==========================================
# 7. STATISTICAL SUMMARY
# ==========================================

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe(include="all").to_string())
df.describe(include="all").to_csv(
    OUTPUT_DIR / "statistical_summary.csv",
    index=False
)

# ==========================================
# 8. NUMERICAL DATA DISTRIBUTION
# ==========================================

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Distribution of {column}")

    save_chart(f"{column}_distribution.png")

# ==========================================
# 9. CATEGORICAL DATA DISTRIBUTION
# ==========================================

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:

    print(f"\nTop values in {column}:")
    print(df[column].value_counts().head(20))

    if df[column].nunique() <= 20:
        plt.figure(figsize=(10, 5))

        df[column].value_counts().head(20).plot(
            kind="bar"
        )

        plt.title(f"Top Categories: {column}")
        plt.xticks(rotation=45, ha="right")
        save_chart(f"{column}_categories.png")
        
# ==========================================
# 10. CORRELATION ANALYSIS
# ==========================================

if len(numeric_columns) > 1:

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        df[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Numerical Feature Correlation")
    plt.tight_layout()
    save_chart("correlation.png")

# ==========================================
# 11. SAVE DATASET REPORT
# ==========================================

with open(OUTPUT_DIR / "dataset_report.txt", "w", encoding="utf-8") as file:

    file.write("ZOMATO DATASET EXPLORATION REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Total Rows: {df.shape[0]}\n")
    file.write(f"Total Columns: {df.shape[1]}\n")
    file.write(f"Dataset Size: {round(df.memory_usage(deep=True).sum() / 1024**2, 2)} MB\n\n")

    file.write("COLUMN INFORMATION\n")
    file.write(column_info.to_string(index=False))

    file.write("\n\nMISSING VALUES\n")
    file.write(missing.to_string())

    file.write("\n\nSTATISTICAL SUMMARY\n")
    file.write(df.describe(include="all").to_string())

    file.write(f"\n\nDuplicate Rows: {df.duplicated().sum()}\n")

print("\n========== DATA EXPLORATION COMPLETED ==========")
