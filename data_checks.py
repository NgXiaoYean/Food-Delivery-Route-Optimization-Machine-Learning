"""
Data-cleaning evidence ("test before you delete")
=================================================
For each cleaning decision in clean_data.py this script measures what it did, so the report can say
"we checked" instead of "we assumed".

  1. Negative coordinates (431 rows) were made positive. If that is a true sign error, those rows should
     then look like every other row (a realistic distance). If the sign were real, distances would be thousands of km.
  2. Rows removed for invalid (~0) coordinates (3,640) and for missing traffic/weather/multi (1,462): are they
     different from the rows we kept (delivery time)? If not, removing them does not bias the simulator.

Run from the project root:   python data_checks.py        Output: data/data_checks.csv
"""
import numpy as np
import pandas as pd

CO = ["Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"]


def haversine_km(a, b, c, d):
    a, b, c, d = map(np.radians, [a, b, c, d])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))


def main():
    raw = pd.read_csv("data/zomato_dataset.csv")
    for c in raw.select_dtypes(include=["object", "string"]).columns:
        raw[c] = raw[c].astype(str).str.strip().replace(r"(?i)^(conditions )?nan$", np.nan, regex=True)
    tt = next(c for c in raw.columns if c.lower().replace(" ", "").startswith("time_taken"))
    raw["t"] = pd.to_numeric(raw[tt].astype(str).str.extract(r"(\d+)")[0])
    neg = (raw[CO] < 0).any(axis=1)
    bad = (raw[CO].abs() <= 1).any(axis=1)
    pos = raw.copy()
    pos[CO] = pos[CO].abs()
    d_fixed = haversine_km(*[pos[c] for c in CO])
    d_as_is = haversine_km(*[raw[c] for c in CO])
    sign_fixed, normal = neg & ~bad, ~neg & ~bad
    miss = (~bad) & raw[["Road_traffic_density", "Weather_conditions", "multiple_deliveries"]].isna().any(axis=1)
    kept = (~bad) & ~miss
    rows = [
        ("negative-coordinate rows", int(sign_fixed.sum()), "", "", ""),
        ("  distance after abs() (km): median / 95th pct", "", round(d_fixed[sign_fixed].median(), 1), round(d_fixed[sign_fixed].quantile(.95), 1), "other rows: %.1f / %.1f" % (d_fixed[normal].median(), d_fixed[normal].quantile(.95))),
        ("  distance if left negative (km): median / 95th pct", "", round(d_as_is[sign_fixed].median(), 1), round(d_as_is[sign_fixed].quantile(.95), 1), "impossible for a city delivery"),
        ("rows removed: invalid (~0) coordinates", int(bad.sum()), "mean time %.1f min" % raw.t[bad].mean(), "kept rows %.1f min" % raw.t[kept].mean(), ""),
        ("rows removed: missing traffic/weather/multi", int(miss.sum()), "mean time %.1f min" % raw.t[miss].mean(), "kept rows %.1f min" % raw.t[kept].mean(), ""),
    ]
    out = pd.DataFrame(rows, columns=["check", "rows", "value_a", "value_b", "note"])
    out.to_csv("data/data_checks.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
