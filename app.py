"""
app.py

Streamlit interface for the House Price Prediction model.

Run from the project root (after train.py has produced a saved model):
    streamlit run app.py

The input form is built dynamically from the schema the model was
trained on (models/metrics.json) and from the value ranges in
data/house_prices.csv, so it keeps working if the dataset is later
swapped for a different one -- no hardcoded column names here.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "house_prices.csv"
TARGET = "price"

st.set_page_config(page_title="House Price Predictor", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_schema():
    numeric_features, categorical_features, metrics = [], [], {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        numeric_features = metrics.get("numeric_features", [])
        categorical_features = metrics.get("categorical_features", [])
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    return numeric_features, categorical_features, metrics, df


def format_inr(amount: float) -> str:
    """Format a rupee amount with comma grouping and a lakh/crore caption."""
    grouped = f"Rs. {amount:,.0f}"
    if amount >= 1_00_00_000:
        caption = f"~{amount / 1_00_00_000:.2f} crore"
    else:
        caption = f"~{amount / 1_00_000:.2f} lakh"
    return grouped, caption


def main():
    st.title("House Price Prediction")
    st.write("Enter the property details below to estimate its price.")

    model = load_model()
    if model is None:
        st.error(f"No trained model found at {MODEL_PATH}. Run `python train.py` first.")
        return

    numeric_features, categorical_features, metrics, df = load_schema()
    if not numeric_features and not categorical_features:
        st.error(
            "No schema found. Run `python train.py` first so "
            "models/metrics.json can be generated."
        )
        return

    if metrics:
        with st.expander("Model performance (on held-out test data)"):
            cols = st.columns(4)
            cols[0].metric("MAE", f"{metrics['MAE']:,.0f}")
            cols[1].metric("RMSE", f"{metrics['RMSE']:,.0f}")
            cols[2].metric("R2", f"{metrics['R2']:.3f}")
            cols[3].metric("Test rows", metrics.get("test_rows", "-"))

    inputs = {}

    if numeric_features:
        st.subheader("Property details")
        num_cols = st.columns(2)
        for i, feature in enumerate(numeric_features):
            col = num_cols[i % 2]
            series = df[feature].dropna() if df is not None and feature in df else None
            if series is not None and not series.empty:
                default = float(series.median())
                min_value = float(max(0, series.min()))
            else:
                default, min_value = 0.0, 0.0
            is_whole_number = series is None or (series.dropna() % 1 == 0).all()
            if is_whole_number:
                inputs[feature] = col.number_input(
                    feature.replace("_", " ").title(),
                    min_value=int(min_value),
                    value=int(default),
                    step=1,
                )
            else:
                inputs[feature] = col.number_input(
                    feature.replace("_", " ").title(),
                    min_value=min_value,
                    value=default,
                    step=1.0,
                )

    if categorical_features:
        st.subheader("Features and location")
        cat_cols = st.columns(2)
        for i, feature in enumerate(categorical_features):
            col = cat_cols[i % 2]
            if df is not None and feature in df:
                options = sorted(df[feature].dropna().unique().tolist())
            else:
                options = ["yes", "no"]
            inputs[feature] = col.selectbox(feature.replace("_", " ").title(), options)

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        grouped, caption = format_inr(prediction)
        st.success(f"Estimated House Price: {grouped}")
        st.caption(caption)


if __name__ == "__main__":
    main()
