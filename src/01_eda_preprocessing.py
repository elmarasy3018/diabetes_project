"""
Step 1: Exploratory Data Analysis + Preprocessing
Pima Indians Diabetes Dataset
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
IMG = "/home/claude/diabetes_project/images"

df = pd.read_csv("/home/claude/diabetes_project/data/diabetes.csv")

print("=" * 60)
print("BASIC INFO")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nMissing values (pandas NaN check):\n{df.isnull().sum()}")

# ---------------------------------------------------------------
# KEY DATA QUALITY ISSUE: biologically impossible zeros
# Glucose, BloodPressure, SkinThickness, Insulin, BMI cannot be 0
# in a living patient -- these zeros represent missing measurements
# ---------------------------------------------------------------
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\n" + "=" * 60)
print("HIDDEN MISSING VALUES (encoded as 0)")
print("=" * 60)
for col in zero_cols:
    n_zero = (df[col] == 0).sum()
    pct = n_zero / len(df) * 100
    print(f"{col:25s}: {n_zero:4d} zeros  ({pct:5.1f}%)")

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe().round(2))

print("\n" + "=" * 60)
print("CLASS BALANCE (Outcome)")
print("=" * 60)
print(df["Outcome"].value_counts())
print(df["Outcome"].value_counts(normalize=True).round(3) * 100)

# ---------------------------------------------------------------
# VISUALIZATIONS (saved to /images for the report)
# ---------------------------------------------------------------
# 1. Class balance
plt.figure(figsize=(5, 4))
sns.countplot(x="Outcome", hue="Outcome", data=df, palette=["#028090", "#F96167"], legend=False)
plt.title("Class Balance: Diabetes Outcome")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.tight_layout()
plt.savefig(f"{IMG}/01_class_balance.png", dpi=150)
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{IMG}/02_correlation_heatmap.png", dpi=150)
plt.close()

# 3. Distributions for the zero-affected columns, split by outcome
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(zero_cols + ["Age"]):
    sns.histplot(data=df, x=col, hue="Outcome", bins=30, kde=True, ax=axes[i], palette=["#028090", "#F96167"])
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig(f"{IMG}/03_feature_distributions.png", dpi=150)
plt.close()

print("\nSaved visualizations to /images")

# =================================================================
# PREPROCESSING
# =================================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

df_clean = df.copy()

# Step 1: Replace biologically impossible zeros with NaN
for col in zero_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)

print("Step 1: Replaced impossible zeros with NaN in:", zero_cols)
print(f"\nMissing values after replacement:\n{df_clean.isnull().sum()}")

# NOTE ON IMPUTATION STRATEGY (important methodological decision):
# We deliberately do NOT impute missing values here using the full dataset
# (e.g. filling with the median grouped by Outcome). Doing so would leak
# information from the target label into the features, and would also leak
# test-set information into training if done before the train/test split.
# Both are data leakage and would produce an overly optimistic, unfair
# accuracy estimate. Instead, missing-value imputation is done LATER,
# inside the modeling pipeline (Step 02), fitted only on the training
# split and then applied unchanged to the test split.
#
# This file keeps the zeros marked as NaN (Step 1 above) and only adds
# descriptive/engineered columns below for exploration purposes -- these
# engineered categorical columns are NOT used as model inputs in Step 02.
print("\nNote: missing-value imputation is deferred to the modeling script (02_modeling.py),")
print("fitted on the training split only, to avoid target leakage and train/test leakage.")

# Step 2: Feature engineering (for exploration/visualization only —
# these categorical columns are not fed into the models in 02_modeling.py)
df_clean["BMI_Category"] = pd.cut(
    df_clean["BMI"], bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)
df_clean["Age_Group"] = pd.cut(
    df_clean["Age"], bins=[0, 30, 45, 60, 100],
    labels=["Young", "Middle-aged", "Senior", "Elderly"]
)
df_clean["Glucose_Category"] = pd.cut(
    df_clean["Glucose"], bins=[0, 100, 126, 300],
    labels=["Normal", "Prediabetic", "Diabetic_Range"]
)
print("\nStep 2: Engineered 3 new categorical features (for exploration): BMI_Category, Age_Group, Glucose_Category")

# Step 3: Duplicate check (none found in this dataset, documented anyway)
n_dup = df_clean.duplicated().sum()
print(f"\nStep 3: Duplicate rows check: {n_dup} found")

# Save cleaned dataset
df_clean.to_csv("/home/claude/diabetes_project/data/diabetes_clean.csv", index=False)
print(f"\nSaved cleaned dataset -> data/diabetes_clean.csv  (shape: {df_clean.shape})")

print("\n" + "=" * 60)
print("FINAL CLEANED DATA SUMMARY")
print("=" * 60)
print(df_clean.describe().round(2))
