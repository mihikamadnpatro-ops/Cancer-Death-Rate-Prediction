# SSDI Project: Predicting County-Level Death Rate

An end-to-end statistical analysis and machine learning pipeline that predicts county-level `deathRate` from socioeconomic and health-related features. The project covers exploratory data analysis, hypothesis testing, feature engineering, regularized regression, and tree-based ensemble models, and compares them on a held-out test set.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Compared](#models-compared)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Output Artifacts](#output-artifacts)
- [Methodology Notes](#methodology-notes)
- [License](#license)

## Overview

**Goal:** Build and compare regression models that predict `deathRate` and identify which factors drive it most.

**Approach:**
1. Explore the data and test key hypotheses statistically.
2. Clean, log-transform, and engineer features (interactions and polynomial terms).
3. Check multicollinearity with VIF.
4. Fit linear, regularized, polynomial, and ensemble models.
5. Compare models on test RMSE and R², then run residual diagnostics and feature importance on the best model.

## Dataset

The notebook expects a file named **`fullData.csv`** (read with `latin-1` encoding) in the project root.

| Feature | Description |
|---|---|
| `deathRate` | **Target variable** |
| `povertyPercent` | Percent of population in poverty |
| `medIncome` | Median income |
| `popEst2015` | Population estimate (2015) |
| `incidenceRate` | Incidence rate |
| `studyCount` | Number of studies |
| `recentTrend`, `recTrend` | Categorical trend (`falling` / `stable` / `rising`), encoded 0/1/2 |
| `fiveYearTrend` | Five-year trend (numeric) |
| `State` | State (dropped before modelling) |

> Add your data source and licensing information here.

## Project Workflow

| Stage | Details |
|---|---|
| **EDA** | Descriptive stats, missing values, target distribution (raw vs. log), correlation heatmap, scatter plots, boxplot by trend |
| **Hypothesis testing** | Two-sample t-test (high vs. low poverty counties); one-way ANOVA across `recentTrend` groups with Tukey HSD post-hoc |
| **Cleaning & encoding** | Ordinal encoding of trend columns, numeric coercion, median imputation, rows with missing target dropped |
| **Feature engineering** | `log1p` on skewed predictors, `log` on the target, 7 interaction terms, 3 squared terms (18 features total) |
| **Multicollinearity** | Variance Inflation Factor (VIF) on core predictors |
| **Split & scaling** | 80/20 train/test split (`random_state=123`); `StandardScaler` fit on train only (a `MinMaxScaler` version is also prepared) |
| **Modelling** | See below |
| **Evaluation** | Test RMSE and R² on the original (back-transformed) scale; 10-fold CV; residual diagnostics; feature importance |

## Models Compared

- Baseline OLS (statsmodels)
- OLS with cubic polynomial terms
- Ridge (alpha chosen by `RidgeCV`)
- Lasso (alpha chosen by `LassoCV`)
- ElasticNet (alpha and `l1_ratio` chosen by `ElasticNetCV`)
- Gradient Boosting Regressor (300 trees, lr=0.05, depth=4)
- Random Forest Regressor (300 trees, depth=8)

Regularization paths for Ridge, Lasso, and ElasticNet are plotted to show how coefficients shrink with alpha. Results are printed as a comparison table at the end of the notebook (Step 22).

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn jupyter
```

### Run

1. Place `fullData.csv` in the project root.
2. Launch Jupyter and open the notebook:

```bash
jupyter notebook ssdi_project.ipynb
```

3. Run all cells (Kernel → Restart & Run All).

## Repository Structure

```
.
├── ssdi_project.ipynb   # Main analysis notebook
├── fullData.csv                      # Input data (not included; add your own)
├── README.md
└── *.png                             # Figures generated when the notebook runs
```

## Output Artifacts

Running the notebook saves these figures to the working directory:

- `deathRate_distribution.png`
- `correlation_heatmap.png`
- `scatter_plots.png`
- `boxplot_trend.png`
- `ridge_coef_path.png`
- `lasso_coef_path.png`
- `enet_coef_path.png`
- `coefficient_paths_combined.png`
- `residual_diagnostics.png`
- `feature_importance.png`

## Methodology Notes

- Scalers are fit on the training set only, to avoid data leakage.
- Models are trained on `log(deathRate)`; metrics are reported after back-transforming with `exp`, so RMSE and R² are on the original scale.
- Missing numeric values are imputed with the column median.
- The notebook uses the `Agg` matplotlib backend, so plots are saved to disk rather than displayed inline.

## Results

> Run the notebook and paste the model comparison table from Step 22 here, for example:
>
> | Model | Test RMSE | Test R² |
> |---|---|---|
> | Baseline OLS | – | – |
> | Gradient Boosting | – | – |

## License

Add your license here (e.g., MIT).
