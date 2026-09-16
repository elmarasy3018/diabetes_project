"""
Step 2: Modeling & Evaluation
Compare multiple classification algorithms for diabetes prediction
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

IMG = "/home/claude/diabetes_project/images"
REPORTS = "/home/claude/diabetes_project/reports"

# Load the RAW dataset (not the EDA "clean" file) -- imputation happens here,
# fitted only on the training split, to avoid any data leakage.
df = pd.read_csv("/home/claude/diabetes_project/data/diabetes.csv")

feature_cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

X = df[feature_cols].copy()
y = df["Outcome"]

# Mark biologically impossible zeros as missing (this step alone introduces
# no leakage -- it is a fixed, deterministic recoding rule, not a statistic
# learned from the data)
for col in zero_as_missing_cols:
    X[col] = X[col].replace(0, np.nan)

print("=" * 60)
print(f"Features used: {feature_cols}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("=" * 60)

# Train/test split FIRST (stratified to preserve class balance), 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Impute + scale, FIT ONLY ON TRAIN, then apply the same fitted transform to test.
# This is the leakage-safe way to handle the missing values found during EDA:
# no information from the test set, and no information from the target label,
# is used to fill in the missing values.
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ---------------------------------------------------------------
# Define candidate models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
}

results = []
roc_data = {}

print("\n" + "=" * 60)
print("BASELINE MODEL COMPARISON (default / lightly-set hyperparameters)")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 5-fold cross-validation on training data for a more robust estimate
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=StratifiedKFold(5), scoring="accuracy")

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "ROC_AUC": auc, "CV_Mean_Accuracy": cv_scores.mean(), "CV_Std": cv_scores.std()
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.4f}   Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}   ROC-AUC: {auc:.4f}")
    print(f"  5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
results_df.to_csv(f"{REPORTS}/model_comparison.csv", index=False)
print("\n" + "=" * 60)
print("SUMMARY TABLE (sorted by ROC-AUC)")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

# ---------------------------------------------------------------
# ROC curve comparison plot
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{IMG}/04_roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Hyperparameter tuning on the best-performing model: Random Forest
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING — Random Forest (GridSearchCV, 5-fold)")
print("=" * 60)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

final_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_best),
    "Precision": precision_score(y_test, y_pred_best),
    "Recall": recall_score(y_test, y_pred_best),
    "F1": f1_score(y_test, y_pred_best),
    "ROC_AUC": roc_auc_score(y_test, y_proba_best),
}
print("\nFinal Tuned Random Forest — Test Set Performance:")
for k, v in final_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=["No Diabetes", "Diabetes"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix — Tuned Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{IMG}/05_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance plot
importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#028090")
plt.title("Feature Importance — Tuned Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{IMG}/06_feature_importance.png", dpi=150)
plt.close()

# Save final metrics + best params to JSON for the report
with open(f"{REPORTS}/final_model_results.json", "w") as f:
    json.dump({
        "best_params": grid.best_params_,
        "cv_roc_auc": grid.best_score_,
        "test_metrics": final_metrics,
        "feature_importance": importances.sort_values(ascending=False).to_dict(),
    }, f, indent=2)

print("\nSaved: reports/model_comparison.csv, reports/final_model_results.json")
print("Saved: images/04_roc_curves.png, 05_confusion_matrix.png, 06_feature_importance.png")
