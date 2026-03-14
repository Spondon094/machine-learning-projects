# 🎓 Student Exam Pass Prediction

A Machine Learning project that predicts whether a student will **Pass or Fail** an exam based on their study habits and academic history, using **Logistic Regression**.

---

## 📌 Problem Statement

A training institute wants to predict whether a student will pass or fail an exam based on their preparation level. Using historical student data, we build a classification model that helps identify at-risk students early.

---

## 📊 Dataset

- **Source:** [Kaggle - Student Performance Dataset](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
- **Total Records:** 10,000 students

### Features Used

| Feature | Description |
|---|---|
| Hours Studied | Number of hours the student studied |
| Previous Scores | Scores from previous tests |
| Sleep Hours | Average hours of sleep per night |

### Target Variable

| Value | Meaning |
|---|---|
| 1 | Pass (Performance Index >= 50) |
| 0 | Fail (Performance Index < 50) |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:**
  - `pandas` — Data loading and manipulation
  - `scikit-learn` — Model building and evaluation
  - `kagglehub` — Dataset download

---

## 🔄 Project Workflow

1. **Load Dataset** — Download dataset using KaggleHub API
2. **Explore Data** — Check shape, data types, and missing values
3. **Define Features & Target** — Select input features and convert Performance Index to binary (Pass/Fail)
4. **Split Dataset** — 80% training, 20% testing (`random_state=42`)
5. **Train Model** — Logistic Regression with `max_iter=1000`
6. **Evaluate Model** — Accuracy score and confusion matrix
7. **Predict** — Predict Pass/Fail for new students

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Accuracy | **97.05%** |
| Correct Predictions | 1941 / 2000 |
| Wrong Predictions | 59 / 2000 |

### Confusion Matrix

```
[[798   28]
 [ 31  1143]]
```

| | Predicted Fail | Predicted Pass |
|---|---|---|
| **Actual Fail** | 798 ✅ | 28 ❌ |
| **Actual Pass** | 31 ❌ | 1143 ✅ |

---

## ⚖️ Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Pass (1) | 5909 | 59.09% |
| Fail (0) | 4091 | 40.91% |

The dataset is well-balanced, ensuring the model learns from both classes effectively.

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Install dependencies:
```python
pip install kagglehub scikit-learn pandas
```
3. Run all cells in order from top to bottom

---

## 📁 Project Structure

```
student-exam-pass-prediction/
│
├── student_exam_prediction.ipynb   # Main Colab notebook
└── README.md                       # Project documentation
```

---

## 🙋 Author

Made with ❤️ as a Machine Learning practice project.
