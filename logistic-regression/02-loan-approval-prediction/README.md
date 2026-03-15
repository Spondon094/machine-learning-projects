# 🏦 Loan Approval Prediction

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** using Logistic Regression.

---

## 📌 Problem Statement

Banks need to decide whether a loan applicant should be approved or rejected based on their financial information. This project builds a **Logistic Regression** classifier to automate that decision.

**Target Variable:** `Loan_Status`
- `Y` → Approved (1)
- `N` → Rejected (0)

---

## 📂 Dataset

- **Source:** [Kaggle - Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **File used:** `train_u6lujuX_CVtuZ9i.csv`
- **Total records:** 614 applicants

### Key Features Used

| Feature | Description |
|---|---|
| `ApplicantIncome` | Monthly income of the applicant |
| `LoanAmount` | Loan amount requested (in thousands) |
| `Credit_History` | Whether the applicant has a good credit history (1 = Yes, 0 = No) |
| `Education` | Graduate (1) / Not Graduate (0) |

---

## 🔧 Project Steps

1. **Load Dataset** — Downloaded via `kagglehub`, loaded using `pandas`
2. **Handle Missing Values** — Numerical columns filled with median, categorical with mode
3. **Encode Categorical Variables** — `LabelEncoder` applied to `Education`, `Married`, `Loan_Status`
4. **Feature Selection** — Selected 4 key features
5. **Train/Test Split** — 80% training, 20% testing (`random_state=42`)
6. **Feature Scaling** — `StandardScaler` applied to normalize feature ranges
7. **Train Model** — `LogisticRegression(max_iter=1000)`
8. **Evaluate Model** — Accuracy score + Confusion Matrix
9. **Predict New Applicant** — Predict approval for a custom input

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | **78.86%** |

### Confusion Matrix

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 18 ✅ | 25 ❌ |
| **Actual Approved** | 1 ❌ | 79 ✅ |

**Observations:**
- Model is strong at predicting approvals (98.7% recall)
- Model struggles with rejections (41.8% recall) — biased toward approval
- Future improvement: address class imbalance with oversampling or threshold tuning

---

## 🧪 Predict a New Applicant

```python
# ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Education']
new_applicant = [[50000, 200, 1, 1]]
new_applicant_scaled = scaler.transform(new_applicant)
prediction = model.predict(new_applicant_scaled)

# Output: Loan Application: APPROVED ✅
```

---

## 🛠️ Tech Stack

- Python 3.12
- pandas
- scikit-learn
- matplotlib
- seaborn
- kagglehub

---

## ⚠️ Known Warning

```
UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
```

This warning appears during new applicant prediction because the input is passed as a plain list instead of a DataFrame. It does not affect prediction results.

**Fix (optional):**
```python
import pandas as pd
new_applicant = pd.DataFrame([[50000, 200, 1, 1]],
                  columns=['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Education'])
```

---

## 📁 File Structure

```
loan-approval-prediction/
│
├── loan_approval_prediction.ipynb   # Main notebook
└── README.md                        # This file
```

---

## 👤 Author

> Part of a personal ML learning repository covering Classification, Regression, and Clustering projects.
