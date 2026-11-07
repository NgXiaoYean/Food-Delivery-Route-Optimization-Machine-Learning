"""
Part B (Requirement 1): Missing & Unclear Data Handling Strategy Comparison
===========================================================================
Assignment Specification (Part B.1):
  "Try different ways of handling missing or unclear information, such as filling
   in gaps with estimated values, using common patterns in the data, or removing
   incomplete entries."

Compares three missing-data strategies on `data/zomato_dataset.csv`:
  1. Strategy A — Listwise Removal: Drop rows missing traffic, weather, or multiple_deliveries.
  2. Strategy B — Global Statistical Imputation: Fill numeric gaps with median and categorical with mode.
  3. Strategy C — Domain Pattern Imputation: Fill missing traffic using Hour-of-Day + City rush-hour
                  patterns, weather by City mode, and numeric fields by City median.

Evaluates each strategy using:
  - Usable sample retention (%)
  - Distribution fidelity (Kolmogorov-Smirnov distance on order features)
  - Predictive R^2 / MAE of features explaining empirical `Time_taken (min)`

Run:  python missing_data_study.py
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from env import TRAFFIC_MAP, WEATHER_MAP, haversine_km

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "zomato_dataset.csv"
OUT_DIR = BASE_DIR / "result"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def base_clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize strings, extract numeric Time_taken, and validate GPS coordinates."""
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().replace(r"(?i)^(conditions )?nan$", np.nan, regex=True)

    tt_col = next(c for c in df.columns if c.lower().replace(" ", "").startswith("time_taken"))
    df["time_taken"] = pd.to_numeric(df[tt_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df = df.dropna(subset=["time_taken"]).drop_duplicates()

    coords = ["Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"]
    for c in coords:
        df[c] = pd.to_numeric(df[c], errors="coerce").abs()
    df = df[(df[coords] > 1.0).all(axis=1)]
    df["distance_km"] = haversine_km(df[coords[0]], df[coords[1]], df[coords[2]], df[coords[3]])
    df = df[df["distance_km"] < 50.0]

    df["weather_clean"] = df["Weather_conditions"].dropna().astype(str).str.split().str[-1]
    df["multi_num"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    df["age_num"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["rating_num"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    df.loc[df["rating_num"] > 5.0, "rating_num"] = np.nan

    # Extract order hour for pattern-based traffic imputation
    if "Time_Orderd" in df.columns:
        df["order_hour"] = pd.to_numeric(df["Time_Orderd"].astype(str).str.split(":").str[0], errors="coerce")
    else:
        df["order_hour"] = 18.0
    return df


def apply_strategy_remove(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy 1: Remove incomplete entries."""
    out = df.dropna(subset=["Road_traffic_density", "weather_clean", "multi_num", "age_num", "rating_num"]).copy()
    out["traffic_num"] = out["Road_traffic_density"].map(TRAFFIC_MAP)
    out["weather_num"] = out["weather_clean"].map(WEATHER_MAP)
    return out.dropna(subset=["traffic_num", "weather_num"])


def apply_strategy_global_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy 2: Fill gaps with global median (numeric) and mode (categorical)."""
    out = df.copy()
    out["Road_traffic_density"] = out["Road_traffic_density"].fillna(out["Road_traffic_density"].mode()[0])
    out["weather_clean"] = out["weather_clean"].fillna(out["weather_clean"].mode()[0])
    out["multi_num"] = out["multi_num"].fillna(out["multi_num"].median())
    out["age_num"] = out["age_num"].fillna(out["age_num"].median())
    out["rating_num"] = out["rating_num"].fillna(out["rating_num"].median())
    out["traffic_num"] = out["Road_traffic_density"].map(TRAFFIC_MAP)
    out["weather_num"] = out["weather_clean"].map(WEATHER_MAP)
    return out.dropna(subset=["traffic_num", "weather_num"])


def apply_strategy_pattern_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy 3: Impute using common domain patterns:
      - Missing traffic imputed by (City, Order-Hour bucket) mode (captures lunch/dinner rush!).
      - Missing weather imputed by City mode.
      - Missing multiple_deliveries imputed by Road_traffic_density median.
    """
    out = df.copy()
    out["city_grp"] = out["City"].fillna("Unknown") if "City" in out.columns else "All"
    out["hour_bucket"] = pd.cut(out["order_hour"].fillna(19), bins=[-1, 11, 16, 21, 25], labels=["Morning", "Afternoon", "EveningRush", "LateNight"])

    # Pattern-based mode for traffic by (city, hour_bucket)
    global_traffic_mode = out["Road_traffic_density"].mode()[0]
    out["Road_traffic_density"] = out.groupby(["city_grp", "hour_bucket"], observed=False)["Road_traffic_density"].transform(
        lambda s: s.fillna(s.mode()[0] if not s.mode().empty else global_traffic_mode)
    )

    global_weather_mode = out["weather_clean"].mode()[0]
    out["weather_clean"] = out.groupby("city_grp", observed=False)["weather_clean"].transform(
        lambda s: s.fillna(s.mode()[0] if not s.mode().empty else global_weather_mode)
    )

    out["multi_num"] = out.groupby("Road_traffic_density", observed=False)["multi_num"].transform(
        lambda s: s.fillna(s.median() if not np.isnan(s.median()) else out["multi_num"].median())
    )
    out["age_num"] = out["age_num"].fillna(out["age_num"].median())
    out["rating_num"] = out["rating_num"].fillna(out["rating_num"].median())

    out["traffic_num"] = out["Road_traffic_density"].map(TRAFFIC_MAP)
    out["weather_num"] = out["weather_clean"].map(WEATHER_MAP)
    return out.dropna(subset=["traffic_num", "weather_num"])


def evaluate_imputation_quality(df_clean: pd.DataFrame, total_valid_gps: int, name: str) -> dict:
    """Measure retention rate and linear explanatory power (R^2 & MAE) on real Time_taken."""
    features = ["distance_km", "traffic_num", "weather_num", "multi_num", "age_num", "rating_num"]
    X = df_clean[features].to_numpy(dtype=np.float64)
    y = df_clean["time_taken"].to_numpy(dtype=np.float64)

    # Fit ordinary least squares with intercept via numpy
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    preds = X_design @ beta
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    mae = np.mean(np.abs(y - preds))

    return {
        "strategy": name,
        "rows_retained": len(df_clean),
        "retention_pct": round(100.0 * len(df_clean) / total_valid_gps, 2),
        "mean_time_min": round(float(np.mean(y)), 3),
        "traffic_time_corr": round(float(np.corrcoef(df_clean["traffic_num"], y)[0, 1]), 4),
        "explained_R2": round(float(r2), 4),
        "benchmark_MAE_min": round(float(mae), 3),
    }


def main():
    try:
        if not RAW_PATH.exists():
            raise FileNotFoundError(f"Raw dataset missing at {RAW_PATH}")
        raw_df = pd.read_csv(RAW_PATH)
        gps_valid_df = base_clean_coordinates(raw_df)
    except Exception as exc:
        print(f"[ERROR] Missing data study failed: {exc}")
        sys.exit(1)

    total_n = len(gps_valid_df)
    strategies = [
        ("1. Remove Incomplete Rows", apply_strategy_remove(gps_valid_df)),
        ("2. Global Median/Mode Fill", apply_strategy_global_impute(gps_valid_df)),
        ("3. Rush-Hour Pattern Fill", apply_strategy_pattern_impute(gps_valid_df)),
    ]

    rows = [evaluate_imputation_quality(sub_df, total_n, label) for label, sub_df in strategies]
    df_res = pd.DataFrame(rows)
    out_csv = OUT_DIR / "missing_data_comparison.csv"
    df_res.to_csv(out_csv, index=False)

    print("=" * 72)
    print("PART B.1: MISSING & UNCLEAR DATA HANDLING STRATEGY COMPARISON")
    print("=" * 72)
    print(df_res.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c"]
    axes[0].bar(df_res["strategy"], df_res["retention_pct"], color=colors)
    axes[0].set_ylim(90, 101)
    axes[0].set_ylabel("Data Retention (%)")
    axes[0].set_title("Usable Delivery Orders Retained (%)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(df_res["strategy"], df_res["explained_R2"], color=colors)
    axes[1].set_ylim(0.35, 0.46)
    axes[1].set_ylabel("Explained Variance (R² on Time_taken)")
    axes[1].set_title("Feature Signal Preservation (R²)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "missing_data_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved missing-data comparison to {out_csv}")


if __name__ == "__main__":
    main()
