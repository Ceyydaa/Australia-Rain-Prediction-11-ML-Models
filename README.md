# 🌦️ Australia Weather Rain Prediction — ML Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-weatherAUS.csv-green"/>
  <img src="https://img.shields.io/badge/Models-11-purple"/>
  <img src="https://img.shields.io/badge/Best%20Model-Random%20Forest-brightgreen"/>
</p>

> Binary classification pipeline that predicts whether it will rain tomorrow in Australia,
> using 11 machine learning models, statistical feature selection, and SMOTE-based class balancing.

---

## 📌 Problem Statement

Given daily weather measurements from 49 Australian weather stations, predict:

**Will it rain tomorrow? (`RainTomorrow` = Yes / No)**

Class distribution after cleaning:
- **No Rain (0):** 43,993 samples — 77.97%
- **Rain (1):**    12,427 samples — 22.03%

---

## 📂 Project Structure

```
australia-rain-prediction/
│
├── LAST_weather_analysis.py        # Full pipeline script  ← run this
├── LAST_weather_analysis.ipynb     # Kaggle notebook version
├── generate_new_confusion_matrices.py  # Optional: alternative CM styles
│
├── LAST_results/
│   ├── phase1/
│   │   ├── tables/
│   │   │   ├── all_metrics.csv
│   │   │   ├── confusion_matrices.csv
│   │   │   ├── cross_validation.csv
│   │   │   ├── mannwhitney_tests.csv
│   │   │   └── normality_tests.csv
│   │   └── images/
│   │       ├── confusion_matrices.png
│   │       ├── roc_curves.png
│   │       ├── metrics_heatmap.png
│   │       └── fscore.png
│   ├── phase2/
│   │   ├── tables/
│   │   │   ├── all_metrics_comparison.csv
│   │   │   ├── confusion_matrices.csv
│   │   │   └── cross_validation.csv
│   │   └── images/
│   │       ├── confusion_matrices.png
│   │       ├── roc_curves.png
│   │       ├── metrics_heatmap.png
│   │       └── fscore.png
│   └── comparison/
│       ├── all_features_vs_selected.csv
│       └── phase_comparison.png
│
└── LAST_models/
    ├── phase1/   # Trained .pkl models — all 21 features
    └── phase2/   # Trained .pkl models — 16 selected features
```

> ⚠️ `weatherAUS.csv` is **not included** in this repository.
> Download from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) and place in project root.

---

## 🔄 Pipeline

```
Raw Data — 145,460 rows × 23 columns
        │
        ▼
1. Data Cleaning
   └── Drop NaN rows → 56,420 rows  (61% removed)
        │
        ▼
2. Categorical Encoding  (LabelEncoder + Data Dictionary)
   └── Date dropped | Location, WindDir, RainToday, RainTomorrow encoded
        │
        ▼
3. Train / Test Split — 80% / 20%  [stratified]
   └── Train: 45,136  |  Test: 11,284
        │
        ▼
4. Statistical Tests — ON ORIGINAL DATA, BEFORE SMOTE ✅
   ├── Shapiro-Wilk + KS  →  All non-normal (justifies Mann-Whitney)
   └── Mann-Whitney U     →  20/21 significant  |  Location (p=0.222) not significant
        │
        ▼
5. SMOTE  (training set only)
   └── No: 35,194 → 35,194  |  Yes: 9,942 → 35,194
        │
        ▼
6. Min-Max Normalization  (fit on train, transform both)
        │
        ├─── PHASE 1 ────────────────────────────────────┐
        │    21 features | 11 models | 5-fold CV          │
        └────────────────────────────────────────────────┘
        │
        ▼
7. Feature Selection  (21 → 16 features)
   ├── RainToday    removed  →  data leakage
   ├── Location     removed  →  Mann-Whitney p = 0.222
   ├── Temp9am      removed  →  correlation r > 0.90
   ├── Temp3pm      removed  →  correlation r > 0.90
   └── Pressure3pm  removed  →  correlation r > 0.90
        │
        ├─── PHASE 2 ────────────────────────────────────┐
        │    16 features | 11 models | 5-fold CV          │
        └────────────────────────────────────────────────┘
        │
        ▼
8. Phase 1 vs Phase 2 Comparison + Save all results
```

---

## 🤖 Models

| # | Model | Type |
|---|-------|------|
| 1 | KNN (k=3) | Instance-based |
| 2 | KNN (k=5) | Instance-based |
| 3 | KNN (k=7) | Instance-based |
| 4 | SVM (RBF kernel) | Kernel |
| 5 | Decision Tree | Tree |
| 6 | Neural Network (MLP 100-50) | Neural |
| 7 | **Random Forest** (100 trees) | Ensemble — Bagging |
| 8 | Gradient Boosting (100 trees) | Ensemble — Boosting |
| 9 | AdaBoost (100 estimators) | Ensemble — Boosting |
| 10 | Logistic Regression | Linear |
| 11 | Naive Bayes | Probabilistic |

---

## 📊 Results

> ⚠️ Accuracy alone is misleading on imbalanced data. **F1-Score** and **ROC-AUC** are the primary metrics.

### Phase 1 — All Features (21)

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | Kappa | ROC-AUC |
|-------|----------|-----------|--------|----------|-------------|-------|---------|
| KNN (k=3) | 0.7741 | 0.4913 | 0.7272 | 0.5864 | 0.7874 | 0.4389 | 0.8132 |
| KNN (k=5) | 0.7730 | 0.4902 | 0.7618 | 0.5965 | 0.7762 | 0.4488 | 0.8393 |
| KNN (k=7) | 0.7701 | 0.4864 | 0.7843 | 0.6004 | 0.7661 | 0.4513 | 0.8505 |
| SVM (RBF) | 0.8080 | 0.5436 | 0.7996 | 0.6472 | 0.8104 | 0.5219 | 0.8876 |
| Decision Tree | 0.7837 | 0.5057 | 0.5698 | 0.5383 | 0.8427 | 0.3946 | 0.7080 |
| Neural Network | 0.8197 | 0.5728 | 0.7123 | 0.6350 | 0.8500 | 0.5171 | 0.8751 |
| **Random Forest** | **0.8560** | **0.6722** | **0.6692** | **0.6715** | **0.9078** | **0.5780** | **0.8973** |
| Gradient Boosting | 0.8322 | 0.6072 | 0.6793 | 0.6406 | 0.8759 | 0.5325 | 0.8829 |
| AdaBoost | 0.8065 | 0.5451 | 0.7348 | 0.6259 | 0.8268 | 0.4993 | 0.8716 |
| Logistic Regression | 0.8064 | 0.5410 | 0.7968 | 0.6444 | 0.8091 | 0.5180 | 0.8860 |
| Naive Bayes | 0.7699 | 0.4856 | 0.7606 | 0.5928 | 0.7725 | 0.4430 | 0.8502 |

### Phase 2 — Selected Features (16)

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| KNN (k=3) | 0.7719 | 0.5852 | 0.8094 |
| KNN (k=5) | 0.7675 | 0.5898 | 0.8328 |
| KNN (k=7) | 0.7689 | 0.5962 | 0.8447 |
| SVM (RBF) | 0.7990 | 0.6363 | 0.8806 |
| Decision Tree | 0.7849 | 0.5404 | 0.7093 |
| Neural Network | 0.8211 | 0.6266 | 0.8663 |
| **Random Forest** | **0.8508** | **0.6577** | **0.8919** |
| Gradient Boosting | 0.8311 | 0.6375 | 0.8799 |
| AdaBoost | 0.8096 | 0.6273 | 0.8682 |
| Logistic Regression | 0.7997 | 0.6342 | 0.8826 |
| Naive Bayes | 0.7700 | 0.5962 | 0.8500 |

### Phase 1 vs Phase 2

| | Phase 1 | Phase 2 | Δ |
|-|---------|---------|---|
| Features | 21 | 16 | −5 |
| Best model | Random Forest | Random Forest | — |
| Best Accuracy | 0.8560 | 0.8508 | −0.0052 |
| Best F1-Score | 0.6715 | 0.6577 | −0.0138 |
| Best ROC-AUC | 0.8973 | 0.8919 | −0.0054 |

Feature selection removed 5 features with minimal performance loss.

---

## ⚠️ Key Methodological Note

**Statistical tests MUST be performed BEFORE SMOTE.**

SMOTE generates synthetic samples that distort the original distribution. Running normality or Mann-Whitney tests after SMOTE produces invalid p-values. This pipeline correctly applies all statistical tests on **original pre-SMOTE training data**.

---

## 🛠️ Installation & Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy joblib
```

```bash
# 1. Download weatherAUS.csv from Kaggle and place in project root
# 2. Run:
python LAST_weather_analysis.py
```

Results → `LAST_results/` | Models → `LAST_models/`

---

## 🗂️ Dataset

**Source:** [Rain in Australia — Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)  
**Author:** Joe Young (jsphyg)  
**License:** CC0 — Public Domain

| | Value |
|-|-------|
| Rows (original) | 145,460 |
| Rows (after cleaning) | 56,420 |
| Features | 23 → 21 (Date dropped) |
| Target | `RainTomorrow` (Yes/No) |
| Class balance | 77.97% No / 22.03% Yes |
| Stations | 49 Australian stations |

> Dataset is **not included** in this repository. Download from Kaggle and place `weatherAUS.csv` in the project root.

---

## 📄 Sunum

[Proje sunumunu görüntüle (PDF)](./AVUSTRALYA_DA_Yagmur_Tahmini.pdf)
