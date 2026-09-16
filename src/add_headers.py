"""Add human-readable column headers to the raw Pima Indians Diabetes CSV."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(DATA / "diabetes_raw.csv", header=None, names=columns)
df.to_csv(DATA / "diabetes.csv", index=False)
print("Shape:", df.shape)
print(df.head())
