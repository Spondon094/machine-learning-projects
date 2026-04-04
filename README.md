# machine-learning-projects

Applied machine learning on real-world datasets, 
with emphasis on rigorous feature engineering, 
temporal validation, and production-oriented design.

## Projects

### 01 — N26: Expenses & Income Prediction
Predicting total income and expenses for 10,000 
N26 bank users from transaction history.

- **Dataset:** 408,546 transactions across 10,000 users
- **Approach:** Temporal split — Feb–May features → Jun–Jul targets
- **Features:** 56 features across 5 groups
- **Models:** 9 models compared — Random Forest & Gradient Boosting win
- **Results:** 36% lift for income, 46% lift for expenses
- **Stack:** Python, Pandas, Scikit-Learn, Matplotlib

📁 [View Project](./n26-expenses-income/)

---

## Roadmap
- [ ] N26 MLOps upgrade — FastAPI + Docker + MLflow
- [ ] More projects counting
