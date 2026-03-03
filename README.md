# Credit Risk & Loan Default Prediction System

This repository contains a Jupyter Notebook implementing an industry-style credit risk and loan default prediction project.

Contents:
- `Credit Risk & Loan Default Prediction.ipynb`: Notebook with data loading, EDA, feature engineering, model training, explainability (SHAP), hyperparameter tuning, imbalance handling (SMOTE), outlier treatment, business insights, and model saving.
- `Loan_default.csv`: Dataset used for the project.

How to run:
1. Create a Python environment (recommend Python 3.8+).
2. Install dependencies: see `requirements.txt`.
3. Open `Credit Risk & Loan Default Prediction.ipynb` in Jupyter and run cells sequentially.

Optional dependencies:
- For explainability and imbalance-handling features, install `shap` and `imbalanced-learn` if you plan to run SHAP visualizations or SMOTE steps:

```bash
pip install shap imbalanced-learn
```

If you prefer not to install these in the base environment, create a separate virtualenv or conda env and install them there before enabling SHAP/SMOTE in the notebook.

Deliverables:
- EDA and visuals in the notebook.
- Model comparison table and final tuned model saved as `final_random_forest.joblib` (artifact files are git-ignored by default).
- Explainability via SHAP and logistic coefficients.
- Simple rule-based decision examples and business insights.

Repository prep notes:
- A `.gitignore` is provided to exclude virtual environments, model artifacts (`*.joblib`), and notebook checkpoints.
- Use `train.py` to reproduce preprocessing and training outside the notebook and to generate artifacts reproducibly.
- Before pushing large artifacts (models, datasets) consider using Git LFS or excluding them from the repo.
 - Removed unused/irrelevant files (for example: `Untitled.ipynb`).

Next steps:
- Optional: create a Flask API to serve the model, or build a Power BI / Tableau dashboard.

Contact:
- Author: Ahmed
