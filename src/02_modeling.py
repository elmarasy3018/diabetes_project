"""
Step 2: Modeling & Evaluation
Compare multiple classification algorithms for diabetes prediction
"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "images"
REPORTS = ROOT / "reports"

# Load the RAW dataset (not the EDA "clean" file) -- imputation happens here,
# fitted only on training data, to avoid any data leakage.
df = pd.read_csv(ROOT / "data" / "diabetes.csv")

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

# Train/test split FIRST (stratified to preserve class balance), 80/20.
# The test set is used ONLY for the final evaluation -- never for choosing
# a model or hyperparameters.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")


# Impute + scale INSIDE a Pipeline. Every time the pipeline is fit (on the
# full training set, or on the training folds during cross-validation), the
# imputer medians and scaler statistics are learned from that data only.
# This keeps the validation folds and the test set fully unseen.
def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------
# Define candidate models + a hyperparameter grid for each
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
}

param_grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    "Decision Tree": {
        "model__max_depth": [3, 4, 5, 6, 8, None],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "K-Nearest Neighbors": {
        "model__n_neighbors": [5, 7, 9, 11, 15, 21, 31],
        "model__weights": ["uniform", "distance"],
    },
    "Support Vector Machine": {
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", 0.01, 0.1],
    },
    "Naive Bayes": {
        "model__var_smoothing": np.logspace(-9, -3, 7),
    },
}


def test_metrics(pipe):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }


results = []
roc_data = {}

print("\n" + "=" * 60)
print("BASELINE MODEL COMPARISON (default / lightly-set hyperparameters)")
print("=" * 60)

for name, model in models.items():
    pipe = make_pipeline(model)

    # 5-fold cross-validation on the TRAINING data -- this is what drives model selection
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "accuracy", "recall"])

    # Test-set scores are reported for reference only
    pipe.fit(X_train, y_train)
    test = test_metrics(pipe)

    results.append({
        "Model": name,
        "CV_ROC_AUC": cv_res["test_roc_auc"].mean(), "CV_ROC_AUC_Std": cv_res["test_roc_auc"].std(),
        "CV_Accuracy": cv_res["test_accuracy"].mean(), "CV_Recall": cv_res["test_recall"].mean(),
        **{f"Test_{k}": v for k, v in test.items()},
    })

    fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
    roc_data[name] = (fpr, tpr, test["ROC_AUC"])

    print(f"\n{name}")
    print(f"  5-fold CV:  ROC-AUC {cv_res['test_roc_auc'].mean():.4f} (+/- {cv_res['test_roc_auc'].std():.4f})"
          f"   Accuracy {cv_res['test_accuracy'].mean():.4f}   Recall {cv_res['test_recall'].mean():.4f}")
    print(f"  Test set:   Accuracy {test['Accuracy']:.4f}   Precision {test['Precision']:.4f}"
          f"   Recall {test['Recall']:.4f}   F1 {test['F1']:.4f}   ROC-AUC {test['ROC_AUC']:.4f}")

results_df = pd.DataFrame(results).sort_values("CV_ROC_AUC", ascending=False)
results_df.to_csv(REPORTS / "model_comparison.csv", index=False)
print("\n" + "=" * 60)
print("SUMMARY TABLE (sorted by 5-fold CV ROC-AUC)")
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
plt.title("ROC Curves — All Models (test set)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(IMG / "04_roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Hyperparameter tuning on the model with the best CV ROC-AUC
# ---------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
baseline_test = {k[len("Test_"):]: v for k, v in results_df.iloc[0].items() if k.startswith("Test_")}

print("\n" + "=" * 60)
print(f"HYPERPARAMETER TUNING — {best_name} (selected by CV ROC-AUC; GridSearchCV, 5-fold)")
print("=" * 60)

grid = GridSearchCV(
    make_pipeline(models[best_name]), param_grids[best_name],
    cv=cv, scoring="roc_auc", n_jobs=-1
)
grid.fit(X_train, y_train)

best_params = {k.removeprefix("model__"): v for k, v in grid.best_params_.items()}
print(f"Best params: {best_params}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}  (untuned: {results_df.iloc[0]['CV_ROC_AUC']:.4f})")

best_pipe = grid.best_estimator_
y_pred_best = best_pipe.predict(X_test)
final_metrics = test_metrics(best_pipe)

print(f"\nFinal Tuned {best_name} — Test Set Performance (untuned baseline in brackets):")
for k, v in final_metrics.items():
    print(f"  {k}: {v:.4f}  [{baseline_test[k]:.4f}]")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=["No Diabetes", "Diabetes"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title(f"Confusion Matrix — Tuned {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(IMG / "05_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance: native importances for tree models, |coefficient| for
# linear models (features are standardized, so magnitudes are comparable),
# otherwise permutation importance on the test set.
fitted_model = best_pipe.named_steps["model"]
if hasattr(fitted_model, "feature_importances_"):
    importance_method = "Impurity-based feature importance"
    raw_importance = fitted_model.feature_importances_
elif hasattr(fitted_model, "coef_"):
    importance_method = "|Coefficient| (standardized features)"
    raw_importance = np.abs(fitted_model.coef_[0])
else:
    importance_method = "Permutation importance (drop in test ROC-AUC)"
    raw_importance = permutation_importance(
        best_pipe, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=42
    ).importances_mean

importances = pd.Series(raw_importance, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#028090")
plt.title(f"Feature Importance — Tuned {best_name}")
plt.xlabel(importance_method)
plt.tight_layout()
plt.savefig(IMG / "06_feature_importance.png", dpi=150)
plt.close()

# Save final metrics + best params to JSON for the report
with open(REPORTS / "final_model_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "selected_model": best_name,
        "selection_criterion": "highest 5-fold CV ROC-AUC on the training set",
        "best_params": best_params,
        "cv_roc_auc": grid.best_score_,
        "test_metrics": final_metrics,
        "untuned_test_metrics": baseline_test,
        "importance_method": importance_method,
        "feature_importance": importances.sort_values(ascending=False).to_dict(),
    }, f, indent=2, default=float)

print("\nSaved: reports/model_comparison.csv, reports/final_model_results.json")
print("Saved: images/04_roc_curves.png, 05_confusion_matrix.png, 06_feature_importance.png")
