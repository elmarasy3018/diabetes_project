"""Add human-readable column headers to the raw Pima Indians Diabetes CSV."""
import pandas as pd

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("/home/claude/diabetes_project/data/diabetes_raw.csv", header=None, names=columns)
df.to_csv("/home/claude/diabetes_project/data/diabetes.csv", index=False)
print("Shape:", df.shape)
print(df.head())
