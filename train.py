"""
train.py

End-to-end training pipeline for House Price Prediction:
load -> inspect -> clean -> EDA -> preprocess -> train -> evaluate -> save.

Run from the project root:
    python train.py

Produces:
    models/house_price_model.pkl   -- the fitted preprocessing + regression pipeline
    reports/figures/*.png          -- EDA and evaluation plots
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; plots are saved to file, not shown

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "house_prices.csv"
MODEL_PATH = ROOT / "models" / "house_price_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"
TARGET = "price"  # change this if your dataset names the price column differently
TEST_SIZE = 0.20
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"\nDropped {before - len(df)} duplicate row(s); {len(df)} rows remain.")
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # "number" catches every numeric dtype (int/float variants). Anything left
    # over is treated as categorical -- more robust than checking for
    # "object" specifically, since recent pandas versions give plain string
    # columns their own "str" dtype instead of "object".
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    print("\nNumerical features  :", numeric_features)
    print("Categorical features:", categorical_features)
    return X, y, numeric_features, categorical_features


def run_eda(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Target distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], kde=True)
    plt.title("House Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_price_distribution.png", dpi=120)
    plt.close()

    # 2. Strongest numeric predictor vs price
    numeric_only = df[numeric_features + [TARGET]].select_dtypes(include="number")
    correlations = numeric_only.corr(numeric_only=True)[TARGET].drop(TARGET).abs()
    scatter_col = correlations.idxmax() if not correlations.empty else numeric_features[0]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=scatter_col, y=TARGET, alpha=0.6)
    plt.title(f"{scatter_col.capitalize()} vs House Price")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_area_vs_price.png", dpi=120)
    plt.close()

    # 3. Correlation heatmap (numeric features only)
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_only.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Numerical Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_correlation_heatmap.png", dpi=120)
    plt.close()

    # 4. Categorical exploration
    cat_col = "location" if "location" in categorical_features else (
        categorical_features[0] if categorical_features else None
    )
    if cat_col:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=cat_col, y=TARGET)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{cat_col.capitalize()} vs House Price")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "04_price_by_category.png", dpi=120)
        plt.close()

    print(f"\nSaved EDA plots to {FIGURES_DIR}")


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])


def evaluate(y_test, y_pred) -> dict:
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    print("\nEvaluation on held-out test set:")
    print(f"  MAE : {metrics['MAE']:,.2f}")
    print(f"  MSE : {metrics['MSE']:,.2f}")
    print(f"  RMSE: {metrics['RMSE']:,.2f}")
    print(f"  R2  : {metrics['R2']:.4f}")
    return metrics


def plot_predictions(y_test, y_pred) -> None:
    # Actual vs predicted
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.6)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_actual_vs_predicted.png", dpi=120)
    plt.close()

    # Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_residual_plot.png", dpi=120)
    plt.close()

    print(f"Saved evaluation plots to {FIGURES_DIR}")


def main():
    df = load_data()
    df = clean_data(df)
    X, y, numeric_features, categorical_features = split_features_target(df)

    run_eda(df, numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTraining rows: {len(X_train)}  |  Testing rows: {len(X_test)}")

    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    print("\nModel trained successfully.")

    y_pred = model.predict(X_test)
    comparison = pd.DataFrame({"Actual Price": y_test.values, "Predicted Price": y_pred})
    print("\nSample predictions:\n", comparison.head(10).to_string(index=False))

    metrics = evaluate(y_test, y_pred)
    plot_predictions(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    metrics["test_rows"] = len(X_test)
    metrics["train_rows"] = len(X_train)
    metrics["numeric_features"] = numeric_features
    metrics["categorical_features"] = categorical_features
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
