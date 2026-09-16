# Predicting Diabetes Risk from Diagnostic Measurements

A Data Mining Algorithms course project: predicting whether a patient is at risk of diabetes using
routine, low-cost diagnostic measurements.

**Course:** Data Mining Algorithms
**Author:** *(add your name)*

---

## 1. Problem Statement

Diabetes is one of the most common chronic diseases worldwide. Early detection meaningfully
improves patient outcomes and reduces long-term treatment costs, but many patients are not
diagnosed until symptoms are already severe.

**Goal:** build a binary classification model that predicts whether a patient is likely to have
diabetes (`Outcome = 1`) or not (`Outcome = 0`), using only 8 routine diagnostic measurements
(number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI,
diabetes pedigree function, and age).

Such a model could support doctors and clinics as a **low-cost, first-pass screening tool** to
flag at-risk patients for further testing. It is explicitly **not** intended to replace a
clinical diagnosis.

## 2. Dataset

- **Name:** Pima Indians Diabetes Dataset
- **Source:** originally from the National Institute of Diabetes and Digestive and Kidney
  Diseases (NIDDK), distributed via the UCI Machine Learning Repository, downloaded here from a
  public GitHub mirror: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
- **Size:** 768 patient records (all female, age 21+, of Pima Indian heritage)
- **Features:** 8 numeric input features + 1 binary target

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Score of diabetes likelihood based on family history |
| `Age` | Age in years |
| `Outcome` | **Target** — 1 = has diabetes, 0 = does not |

Raw file: [`data/diabetes_raw.csv`](data/diabetes_raw.csv) (no headers) →
headers added in [`data/diabetes.csv`](data/diabetes.csv).

## 3. Data Preprocessing

Full step-by-step details, code, and output are in
[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb). Summary:

1. **Detected hidden missing values.** `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
   and `BMI` contain the value `0`, which is biologically impossible for a living patient — these
   are missing measurements, not real readings.
   - `Insulin`: 48.7% missing
   - `SkinThickness`: 29.6% missing
   - `BloodPressure`: 4.6% missing
   - `BMI`: 1.4% missing
   - `Glucose`: 0.7% missing
2. **Checked for duplicate rows** — none found.
3. **Checked class balance** — 65.1% no-diabetes vs. 34.9% diabetes (moderately imbalanced;
   handled by reporting Precision/Recall/F1/ROC-AUC alongside accuracy, not by resampling).
4. **Correlation analysis** — `Glucose` (0.47) and `BMI` (0.29) correlate most strongly with the
   target, consistent with clinical expectations.
5. **Feature engineering (exploratory only):** `BMI_Category`, `Age_Group`, and
   `Glucose_Category` were derived for visualization, confirming diabetes rate rises with glucose
   band, BMI band, and age group. These engineered columns are **not** used as model inputs.
6. **Missing value imputation — leakage-safe.** Missing values are filled using
   `SimpleImputer(strategy="median")`, **fit only on the training split**, then applied unchanged
   to the test split.

   **Important methodological note:** an earlier version of this pipeline imputed missing values
   using the median computed *per target class* (grouped by `Outcome`). This was identified as a
   **target leakage** bug — it directly encodes the answer into the feature for the ~49% of
   `Insulin` values that were missing, producing an artificially inflated accuracy (~86%). This
   was corrected to the leakage-safe approach described above, giving an honest, lower, but
   trustworthy accuracy (~75%). See Section 6 of the notebook for the full explanation.
7. **Feature scaling** — `StandardScaler`, fit only on the training split.

## 4. Data Mining Technique & Algorithm Selection

**Technique:** supervised classification (binary), since the goal is predicting a labeled target.

**Algorithms compared:**

| Algorithm | Rationale |
|---|---|
| Logistic Regression | Simple, interpretable linear baseline |
| Decision Tree | Interpretable, non-linear, handles feature interactions |
| Random Forest | Ensemble of trees; typically more accurate & robust to noise |
| K-Nearest Neighbors | Simple, non-parametric, works well for "similar patient" reasoning |
| Support Vector Machine | Effective non-linear decision boundary for moderate-dimension numeric data |
| Naive Bayes | Fast probabilistic baseline |

All six were evaluated with reasonable default settings; the best performer (**Random Forest**,
by ROC-AUC) was then tuned via `GridSearchCV` (5-fold cross-validation) over `n_estimators`,
`max_depth`, `min_samples_split`, and `min_samples_leaf`.

## 5. Tool Used

**Python 3.12** with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, run inside a
Jupyter notebook. Chosen over a GUI data mining tool (e.g. RapidMiner/Weka) for full control over
the preprocessing pipeline — critical for correctly avoiding the leakage issue described above —
and for reproducibility.

## 6. Results

### Baseline model comparison (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV Accuracy |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.747 | 0.660 | 0.574 | 0.614 | **0.816** | 0.770 ± 0.043 |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 | 0.782 ± 0.012 |
| Support Vector Machine | 0.740 | 0.652 | 0.556 | 0.600 | 0.796 | 0.769 ± 0.018 |
| K-Nearest Neighbors | 0.734 | 0.633 | 0.574 | 0.602 | 0.788 | 0.764 ± 0.035 |
| Naive Bayes | 0.701 | 0.567 | 0.630 | 0.596 | 0.765 | 0.759 ± 0.008 |
| Decision Tree | 0.760 | 0.639 | 0.722 | 0.678 | 0.762 | 0.723 ± 0.019 |

Full CSV: [`reports/model_comparison.csv`](reports/model_comparison.csv)

### Final tuned model: Random Forest

- **Best hyperparameters:** `max_depth=6, min_samples_leaf=4, min_samples_split=2, n_estimators=300`
- **5-fold CV ROC-AUC:** 0.841
- **Test set:** Accuracy 0.747 · Precision 0.683 · Recall 0.519 · F1 0.589 · **ROC-AUC 0.806**

Full metrics + feature importances: [`reports/final_model_results.json`](reports/final_model_results.json)

**Top predictive features:** Glucose (37%), BMI (17%), Age (12%), DiabetesPedigreeFunction (9%),
Insulin (9%) — see [`images/06_feature_importance.png`](images/06_feature_importance.png).

### Discussion

- The model performs meaningfully better than random guessing (ROC-AUC 0.81) and its top features
  match known clinical risk factors for diabetes.
- **Recall on the diabetes class (0.52) is the weakest metric** — of 54 diabetic patients in the
  test set, 26 were missed (false negatives). In a real screening context this is the most
  important number to improve, since a missed diagnosis is costlier than a false alarm.
- The corrected (leakage-safe) accuracy of ~75% is intentionally reported instead of the
  ~86% obtained by the earlier, flawed pipeline — see Section 3, point 6.

### Limitations

- Dataset is limited to female patients of Pima Indian heritage — results may not generalize to
  other populations.
- ~49% of `Insulin` and ~30% of `SkinThickness` values were imputed, adding noise to two
  potentially informative features.
- 768 records is a small sample for machine learning.
- Not validated for real clinical/screening use.

### Future Work

- Try Gradient Boosting / XGBoost.
- Use more advanced, still leakage-safe imputation (e.g. `IterativeImputer`).
- Apply class-imbalance techniques (SMOTE, class weighting) specifically to improve recall on the
  diabetic class.
- Validate on a larger, more diverse dataset before any real-world use.

## Project Structure

```
diabetes_project/
├── README.md                      <- this file
├── requirements.txt
├── data/
│   ├── diabetes_raw.csv           <- original download, no headers
│   ├── diabetes.csv               <- headers added
│   └── diabetes_clean.csv         <- exploration version (NaNs marked, engineered columns)
├── notebooks/
│   └── analysis.ipynb             <- full, executed, end-to-end analysis (start here)
├── src/
│   ├── add_headers.py
│   ├── 01_eda_preprocessing.py
│   ├── 02_modeling.py
│   └── build_notebook.py
├── images/                        <- all charts, generated by the scripts/notebook
└── reports/
    ├── 01_eda_output.txt
    ├── 02_modeling_output.txt
    ├── model_comparison.csv
    └── final_model_results.json
```

## How to Reproduce

```bash
pip install -r requirements.txt
python src/add_headers.py
python src/01_eda_preprocessing.py
python src/02_modeling.py
# or, to see the full narrated analysis:
jupyter notebook notebooks/analysis.ipynb
```
