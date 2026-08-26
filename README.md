# House Price Prediction

End-to-end Linear Regression pipeline that predicts house prices from property
features (area, bedrooms, bathrooms, amenities, location, and more), with a
Streamlit app for interactive predictions.

## Dataset

Kaggle isn't reachable from the environment this project was built in, so
`generate_dataset.py` creates a synthetic-but-realistic dataset
(`data/house_prices.csv`, 1200+ rows) with the same shape as the well-known
Kaggle "Housing Price Prediction" dataset: numeric features (`area`,
`bedrooms`, `bathrooms`, `stories`, `parking`), yes/no amenity flags
(`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`,
`prefarea`), a `furnishingstatus` category, and a `location` column across
eight Indian cities. It also seeds in a few missing values, duplicate rows,
and outliers on purpose, so the cleaning and imputation code in `train.py`
has something real to do instead of running on already-perfect data.

The city price-per-square-foot multipliers and amenity "bonuses" in
`generate_dataset.py` are illustrative assumptions for a self-contained,
reproducible dataset (fixed random seed) -- not real market data.

**To use a real dataset instead:** download a housing dataset from Kaggle and
replace `data/house_prices.csv`. `train.py` and `app.py` both detect numeric
vs. categorical columns dynamically, so nothing else needs to change unless
the price column is named something other than `price` (update the `TARGET`
constant at the top of `train.py`).

Downloading (repo & dataset)

If you want to run this project with a real Kaggle dataset or simply obtain
this repository locally, follow the steps below.

1) Downloading the repository

- Clone via HTTPS (recommended):

```bash
git clone https://github.com/Suchismita185/House-Price-Prediction.git
cd House-Price-Prediction
```

- Clone via SSH (if you have an SSH key configured):

```bash
git clone git@github.com:Suchismita185/House-Price-Prediction.git
cd House-Price-Prediction
```

- Quick download (ZIP) from the web UI: visit
`https://github.com/Suchismita185/House-Price-Prediction` and click **Code →
Download ZIP**; then unzip and open the project folder.

2) Downloading a dataset from Kaggle (optional)

If you prefer to use a Kaggle dataset instead of the provided synthetic CSV,
there are two common workflows below. After downloading, make sure the CSV is
saved as `data/house_prices.csv` (file name must match the default `TARGET`
unless you change `TARGET` in `train.py`).

- Using the Kaggle CLI (recommended for automation):

  1. Install the Kaggle CLI: `pip install kaggle`.
  2. Create an API token on Kaggle (Account → API → Create New API Token).
     This downloads a `kaggle.json` file. Place it at `~/.kaggle/kaggle.json`
     and set permissions: `chmod 600 ~/.kaggle/kaggle.json`.
  3. Download a dataset. Examples:

     - Kaggle competition (House Prices - Advanced Regression Techniques):

       ```bash
       kaggle competitions download -c house-prices-advanced-regression-techniques \
         -f train.csv -p data/ --unzip
       mv data/train.csv data/house_prices.csv
       ```

     - Generic dataset by slug (replace `owner/dataset-name`):

       ```bash
       kaggle datasets download -d owner/dataset-name -p data/ --unzip
       # then rename the relevant CSV to data/house_prices.csv
       ```

  4. Verify the file exists: `ls -l data/house_prices.csv`.

- Manual download (browser):

  1. On Kaggle, open the dataset page you want and download the CSV files.
  2. Place the chosen CSV into the project's `data/` directory and rename it
     to `house_prices.csv` if needed.

Notes and caveats

- Column names and target: The code expects the target column to be named
  `price` by default. If your dataset uses a different target column name,
  change the `TARGET` constant at the top of `train.py`.
- Secrets: Do NOT commit `kaggle.json` or other credentials into the repo.
  Keep API tokens out of version control.
- If your CSV has a different schema, `train.py` detects numeric vs
  categorical columns dynamically; however, inspect the columns first and
  adjust any column name mismatches as needed.

## Project structure

```
house-price-prediction/
├── data/
│   └── house_prices.csv
├── notebooks/
│   └── house_price_analysis.ipynb   # exploration, mirrors train.py, pre-executed
├── models/
│   ├── house_price_model.pkl        # fitted preprocessing + regression pipeline
│   └── metrics.json                 # last training run's metrics + schema
├── reports/figures/                 # EDA and evaluation plots (see below)
├── generate_dataset.py              # builds data/house_prices.csv
├── train.py                         # load -> clean -> EDA -> train -> evaluate -> save
├── compare_models.py                # Ridge / Random Forest comparison + cross-validation
├── app.py                           # Streamlit prediction UI
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running it

```bash
python generate_dataset.py    # (optional) regenerate data/house_prices.csv
python train.py               # train, evaluate, save the model + plots
python compare_models.py      # (optional) Ridge / Random Forest + cross-validation
streamlit run app.py          # interactive prediction UI
```

Or open `notebooks/house_price_analysis.ipynb` for the same workflow with
inline plots and narrative explanation.

## Results

Linear Regression, evaluated on a held-out 20% test split (241 rows):

| Metric | Value |
|---|---|
| MAE  | 3,608,940.80 |
| MSE  | 21,250,953,478,188.25 |
| RMSE | 4,609,875.65 |
| R²   | 0.8607 |

MAE and RMSE are in the same units as `price` (INR). R² of 0.86 means the
model explains about 86% of the price variance in the test set; the
remaining gap comes from the non-linear and noise components baked into the
synthetic price formula, by design, so linear regression cannot fit it
perfectly.

**Actual vs. predicted:**

![Actual vs predicted](reports/figures/05_actual_vs_predicted.png)

**Residuals** (roughly centered on zero with no strong funnel or curve,
which is what you want to see from a linear model):

![Residual plot](reports/figures/06_residual_plot.png)

### Model comparison (`compare_models.py`)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 3,608,940.80 | 4,609,875.65 | 0.8607 |
| Ridge (alpha=1.0) | 3,599,683.49 | 4,605,505.58 | 0.8610 |
| Random Forest | 2,540,972.74 | 3,426,950.00 | 0.9230 |

Random Forest scores noticeably higher, which is expected -- it can capture
the non-linear relationship between area and price that a straight line
can't. Linear Regression remains the required baseline for this project.

**5-fold cross-validation** (Linear Regression, R² per fold):
`[0.7582, 0.8771, 0.8688, 0.8615, 0.8768]` -- mean **0.8485**, std **0.0455**.
Performance is reasonably consistent across folds, with one softer fold that
would be worth investigating (e.g. checking whether it happens to contain
more of the seeded outliers).

### More EDA plots

| Price distribution | Area vs. price | Correlation heatmap | Price by location |
|---|---|---|---|
| ![](reports/figures/01_price_distribution.png) | ![](reports/figures/02_area_vs_price.png) | ![](reports/figures/03_correlation_heatmap.png) | ![](reports/figures/04_price_by_category.png) |

## Implementation notes

- **Preprocessing lives inside the pipeline** (`ColumnTransformer` +
  `SimpleImputer` + `OneHotEncoder`, wrapped with `LinearRegression` in one
  `Pipeline`), so imputation medians/modes and one-hot categories are learned
  from the training split only -- no leakage from the test set.
- **Numeric/categorical detection is dtype-based, not name-based**
  (`select_dtypes(include="number")`, everything else treated as
  categorical). This was checked against the actual pandas version in the
  build environment (pandas 3.0), which gives plain string columns their own
  `"str"` dtype instead of `"object"` -- code that only checks for
  `"object"` silently drops every categorical column on newer pandas, so the
  dtype-agnostic version here is the safer default.
- `app.py` builds its input form from `models/metrics.json` (the schema the
  model was actually trained on) and from the value ranges in
  `data/house_prices.csv`, rather than hardcoding field names, so it keeps
  working if the dataset is swapped.

## Possible extensions

- Feature engineering (price-per-square-foot, property age if available)
- Hyperparameter tuning for Ridge/Random Forest (`GridSearchCV`)
- A held-out final test set kept untouched through model selection
- Investigate the weaker cross-validation fold and the largest residuals
