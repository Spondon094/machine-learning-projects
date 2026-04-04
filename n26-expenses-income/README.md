# N26: Expenses & Income Prediction

Predicting future total income and total expenses 
for 10,000 N26 bank users using 4 months of 
transaction history.

## Problem
Given transaction records and demographic profiles,
build regression models to predict each user's 
total income and total expenses in a future 2-month 
window.

## Dataset
| File | Description |
|---|---|
| `transactions_data_training.csv` | 408,546 transactions across 10,000 users |
| `user_profile.csv` | Demographics — age, country, employment, income band |
| `transaction_types.csv` | Transaction type codes and flow direction |
| `mcc_group_definition.csv` | Merchant category codes (17 categories) |

## Approach
- **Temporal split** — Feb–May 2016 features → Jun–Jul 2016 targets
- **No data leakage** — target window never seen during feature engineering
- **56 features** across 5 groups: counts, amounts, MCC categories, temporal, behavioural
- **9 models compared** — from naive baselines to gradient boosted trees

## Results
| Target | Model | Test MAE | Test R² | Lift vs Baseline |
|---|---|---|---|---|
| Income | Random Forest | 218.6 | 0.367 | 36% |
| Expenses | Gradient Boosting | 233.4 | 0.435 | 46% |

## Key Findings
- Tree models dominated — linear models scored negative R²
- `n_transactions` and `age` are the strongest income predictors
- `city_tier` drives expenses more than income
- `spending_slope` (temporal feature) ranked in top 15 for both targets

## Stack
Python · Pandas · Scikit-Learn · Matplotlib

## Roadmap
- [ ] FastAPI prediction endpoint
- [ ] Docker containerisation
- [ ] MLflow experiment tracking
- [ ] AWS deployment
