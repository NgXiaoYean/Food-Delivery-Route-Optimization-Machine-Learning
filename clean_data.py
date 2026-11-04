import numpy as np
import pandas as pd

path = "./data/zomato_dataset.csv" 
df = pd.read_csv(path)
log = [("Raw data", len(df), "")]

def record(step, note=""):
    log.append((step, len(df), note))
    print(f"{step:<45} rows left: {len(df)}  {note}")
 
 
# ---- 1. Tidy text: strip spaces and turn 'NaN' strings into real missing values
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip().replace(r"(?i)^(conditions )?nan$", np.nan, regex=True)
record("1. Strip text / fix 'NaN' strings", "text columns had stray spaces and 'NaN ' as text")
 
# ---- 2. Target column -> integer minutes
tt = next(c for c in df.columns if c.lower().replace(" ", "").startswith("time_taken"))
df[tt] = pd.to_numeric(df[tt].astype(str).str.extract(r"(\d+)")[0])
df = df.dropna(subset=[tt])
record("2. Target as numeric minutes", f"column: {tt}")
 
# ---- 3. Duplicates
df = df.drop_duplicates()
record("3. Remove duplicate rows")
 
# ---- 4. Coordinates
coords = ["Restaurant_latitude", "Restaurant_longitude",
          "Delivery_location_latitude", "Delivery_location_longitude"]
neg = (df[coords] < 0).any(axis=1).sum()
print(f"   (info) rows with a negative coordinate: {neg}  -> treated as sign errors, made positive")
df[coords] = df[coords].abs()  # DECISION: check this against a map for your cities, and justify it in the report
df = df[(df[coords] > 1).all(axis=1)]
record("4. Remove invalid (~0) coordinates", "0.0 means GPS was not recorded")
 
 
# ---- 5. Distance feature + impossible distances
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
 
 
df["distance_km"] = haversine_km(df[coords[0]], df[coords[1]], df[coords[2]], df[coords[3]])
print("   (info) distance_km summary:\n", df["distance_km"].describe().round(2).to_string())
df = df[df["distance_km"] < 50]
record("5. Remove distances >= 50 km", "a food delivery cannot realistically be this far")
 
# ---- 6. Rows missing fields the environment needs (traffic, weather, multi-delivery)
df = df.dropna(subset=["Road_traffic_density", "Weather_conditions", "multiple_deliveries"])
record("6. Drop rows missing traffic/weather/multi", "env needs these; ~600-1000 each, small loss")
 
# ---- 7. Invalid ratings and missing age/rating (not used by env -> fill, don't drop)
if "Delivery_person_Ratings" in df.columns:
    df = df[(df["Delivery_person_Ratings"].isna()) | (df["Delivery_person_Ratings"] <= 5)]
    record("7a. Remove ratings > 5", "ratings scale is 1-5")
for c in ["Delivery_person_Age", "Delivery_person_Ratings"]:
    df[c + "_was_missing"] = df[c].isna()
    df[c] = df[c].fillna(df[c].median())  # median is robust to outliers
record("7b. Fill missing age/rating with median", "flag columns *_was_missing kept for transparency")
 
# ---- 8. Outliers in delivery time: FLAG only (don't delete, they are real deliveries)
q1, q3 = df[tt].quantile([0.25, 0.75])
iqr = q3 - q1
df["time_outlier"] = (df[tt] < q1 - 1.5 * iqr) | (df[tt] > q3 + 1.5 * iqr)
print(f"   (info) delivery-time outliers flagged (IQR rule): {df['time_outlier'].sum()}")
record("8. Flag time outliers (kept)")
 
# ---- Save
df.to_csv("./data/dataset_clean.csv", index=False)
pd.DataFrame(log, columns=["step", "rows_after", "note"]).to_csv("./data/cleaning_log.csv", index=False)
print("\nSaved dataset_clean.csv and cleaning_log.csv")
print("Remaining missing values:\n", df.isna().sum()[df.isna().sum() > 0].to_string() or "none")