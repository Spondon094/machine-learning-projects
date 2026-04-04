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
