"""
generate_dataset.py

Creates a synthetic-but-realistic housing dataset for the House Price
Prediction project.

Why synthetic data: a live Kaggle download requires Kaggle account
credentials and Kaggle isn't reachable from this build environment, so
this script generates a dataset that mirrors the structure of the
well-known Kaggle "Housing Price Prediction" dataset (numeric features
area / bedrooms / bathrooms / stories / parking, several yes-no amenity
flags, a furnishing status, and a target price) plus a `location`
column, since the implementation guide's own example code references
`location`.

The relationships below (which city costs more per square foot, how
much an extra bathroom is "worth", etc.) are illustrative assumptions
for a self-contained teaching dataset, not real market data. To use a
real dataset instead, drop a Kaggle CSV into data/house_prices.csv --
train.py detects numeric/categorical columns dynamically, so only the
TARGET name at the top of train.py needs to change if the real
dataset's price column is named differently.

Running this script regenerates data/house_prices.csv deterministically
(fixed random seed = reproducible output).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 1200

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "house_prices.csv"

CITY_PRICE_PER_SQFT = {
    "Mumbai": 16000,
    "Delhi": 11000,
    "Bangalore": 9500,
    "Pune": 7800,
    "Hyderabad": 6800,
    "Chennai": 7200,
    "Kolkata": 6200,
    "Ahmedabad": 5600,
}


def generate_raw_features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    cities = list(CITY_PRICE_PER_SQFT.keys())
    city_weights = [0.16, 0.14, 0.15, 0.11, 0.11, 0.11, 0.11, 0.11]
    location = rng.choice(cities, size=n, p=city_weights)

    area = rng.gamma(shape=6.0, scale=380, size=n) + 450
    area = np.clip(area, 500, 7500).round(0)

    bedrooms = np.clip(np.round(area / 750 + rng.normal(0, 0.6, n)), 1, 6).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.7 + rng.normal(0, 0.5, n)), 1, 5).astype(int)
    stories = np.clip(np.round(rng.normal(1.8, 0.8, n)), 1, 4).astype(int)
    parking = np.clip(np.round(rng.normal(1.0, 0.9, n)), 0, 3).astype(int)

    def yes_no(p):
        return np.where(rng.random(n) < p, "yes", "no")

    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": yes_no(0.85),
        "guestroom": yes_no(0.30),
        "basement": yes_no(0.35),
        "hotwaterheating": yes_no(0.15),
        "airconditioning": yes_no(0.40),
        "prefarea": yes_no(0.25),
        "furnishingstatus": rng.choice(
            ["furnished", "semi-furnished", "unfurnished"],
            size=n, p=[0.30, 0.40, 0.30],
        ),
        "location": location,
    })
    df["_price_per_sqft"] = df["location"].map(CITY_PRICE_PER_SQFT)
    return df


def generate_price(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    n = len(df)
    base = df["area"] * df["_price_per_sqft"]

    bonus = (
        df["bedrooms"] * 150_000
        + df["bathrooms"] * 200_000
        + df["stories"] * 120_000
        + df["parking"] * 80_000
        + (df["mainroad"] == "yes") * 150_000
        + (df["guestroom"] == "yes") * 100_000
        + (df["basement"] == "yes") * 130_000
        + (df["hotwaterheating"] == "yes") * 90_000
        + (df["airconditioning"] == "yes") * 250_000
        + (df["prefarea"] == "yes") * 300_000
        + df["furnishingstatus"].map(
            {"furnished": 220_000, "semi-furnished": 90_000, "unfurnished": 0}
        )
    )

    raw_price = base + bonus
    # Mild non-linear wrinkle so a straight line can't fit perfectly (keeps R2 realistic)
    raw_price = raw_price * (1 + 0.05 * np.sin(df["area"] / 900))
    # Heteroscedastic noise: bigger/pricier houses have noisier prices, like real markets
    noise = rng.normal(0, 1, n) * (raw_price * 0.10)
    price = raw_price + noise
    return np.clip(price, 500_000, None).round(-3)


def inject_outliers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    idx = df.sample(n=3, random_state=int(rng.integers(0, 10_000))).index
    df.loc[idx[0], ["area", "price"]] = [7200, df["price"].max() * 1.6]
    df.loc[idx[1], ["area", "price"]] = [620, df["price"].min() * 0.6]
    df.loc[idx[2], "price"] = df.loc[idx[2], "price"] * 2.1
    return df


def inject_duplicates(df: pd.DataFrame, rng: np.random.Generator, n_dupes: int = 6) -> pd.DataFrame:
    dupe_rows = df.sample(n=n_dupes, random_state=int(rng.integers(0, 10_000)))
    return pd.concat([df, dupe_rows], ignore_index=True)


def inject_missing_values(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for col, frac in [("area", 0.02), ("bathrooms", 0.015), ("furnishingstatus", 0.02)]:
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan
    return df


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = generate_raw_features(rng, N_SAMPLES)
    df["price"] = generate_price(df, rng)
    df = df.drop(columns="_price_per_sqft")

    df = inject_outliers(df, rng)
    df = inject_duplicates(df, rng)
    df = inject_missing_values(df, rng)

    cols = ["price"] + [c for c in df.columns if c != "price"]
    df = df[cols]
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns to {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
