# 💳 N26: Expenses & Income Prediction

> Predicting future total income and total expenses for 10,000 
> N26 bank users using 4 months of transaction history.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.2+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🎯 Problem
Given transaction records and demographic profiles, build 
regression models to predict each user's total income and 
total expenses in a future 2-month window.

---

## 📂 Dataset
| File | Description |
|---|---|
| `transactions_data_training.csv` | 408,546 transactions across 10,000 users |
| `user_profile.csv` | Demographics — age, country, employment, income band |
| `transaction_types.csv` | Transaction type codes and flow direction |
| `mcc_group_definition.csv` | Merchant category codes (17 categories) |

---

## 🏗️ Approach

### Temporal Split Design

- ✅ No data leakage — target window never seen during feature engineering
- ✅ Mirrors real production ML — past behaviour predicts future spending
- ✅ Churn-inclusive — 17% inactive users kept as zero targets

### Feature Engineering (56 features across 5 groups)
| Group | Features |
|---|---|
| 📊 Counts | Transaction volume by type and direction |
| 💰 Amounts | Sum, mean, std, median by inflow/outflow |
| 🏪 MCC Categories | Spend across 17 merchant categories |
| 📅 Temporal | Activity span, txn per day, active months |
| 🧠 Behavioural | Spending slope, in/out ratio, weekend ratio |

---

## 📈 Results
| Target | Model | Test MAE | Test R² | Lift vs Baseline |
|---|---|---|---|---|
| 💚 Income | Random Forest | 218.6 | 0.367 | +36% |
| ❤️ Expenses | Gradient Boosting | 233.4 | 0.435 | +46% |

---

## 🔍 Key Findings
- 🌲 **Tree models dominated** — linear models scored negative R²
- 🔢 **`n_transactions` + `age`** are the strongest income predictors
- 🏙️ **`city_tier`** drives expenses more than income
- 📉 **`spending_slope`** (temporal feature) ranked top 15 for both targets
- 👥 **Demographics add real signal** — `income_band`, `employment_status`, `city_tier`

---

## ⚠️ Limitations
- Short time horizon — 4 months of features, 2 months of targets
- Model under-predicts high spenders (regression toward mean)
- Stationarity assumed across the split boundary

---

## 🛠️ Stack
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.2+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?logo=matplotlib)

---

## 🗺️ Roadmap
- [x] EDA & feature engineering
- [x] Model training & evaluation
- [x] Feature importance analysis
- [ ] FastAPI prediction endpoint
- [ ] Docker containerisation
- [ ] MLflow experiment tracking
- [ ] AWS deployment

🚀 Production service: [n26-ml-service]
(https://github.com/Spondon094/n26-ml-service)
