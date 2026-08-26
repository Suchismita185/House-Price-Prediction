"""
compare_models.py

Optional extension (see the implementation guide's "Improve the Model" /
"Compare Models" / "Cross-Validation" sections). Linear Regression is the
required baseline used by train.py and app.py; this script checks whether
a different estimator captures the data better, and how consistent
Linear Regression's performance is across folds.

Run from the project root:
    python compare_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "house_prices.csv"
TARGET = "price"
RANDOM_STATE = 42


def load_clean_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return X, y, numeric_features, categorical_features


def make_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])


def compare_models(X, y, numeric_features, categorical_features):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }

    print("Model comparison (single 80/20 split):\n")
    results = []
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        results.append((name, mae, rmse, r2))

        print(f"{name}")
        print(f"  MAE : {mae:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  R2  : {r2:.4f}\n")

    return results


def cross_validate_baseline(X, y, numeric_features, categorical_features):
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])

    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print("5-fold cross-validation (Linear Regression, scoring=R2):")
    print("  Fold R2 scores:", np.round(scores, 4).tolist())
    print(f"  Mean R2: {scores.mean():.4f}  (std: {scores.std():.4f})")


def main():
    X, y, numeric_features, categorical_features = load_clean_data()
    compare_models(X, y, numeric_features, categorical_features)
    cross_validate_baseline(X, y, numeric_features, categorical_features)


if __name__ == "__main__":
    main()
