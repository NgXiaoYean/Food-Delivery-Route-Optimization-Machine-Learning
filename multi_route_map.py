"""Show simulated multi-order grid routes on an illustrative street basemap.

The environment has no GPS observations for the combined trip. Dataset mode
anchors the drawing at the first selected order's restaurant; all routes and
customer pins remain projected grid coordinates, not recorded road journeys.
"""
from __future__ import annotations

import math
from pathlib import Path

import folium
import pandas as pd

from env import TRAFFIC_MAP, WEATHER_MAP, haversine_km

FALLBACK_CENTER = (23.2599, 77.4126)  # Illustrative Bhopal centre for custom trips.
COORDS = (
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
)
COLORS = {
    "Greedy (nearest first)": "#94a3b8",
    "Planner (oracle)": "#334155",
    "D3QN single model": "#2563eb",
    "D3QN ensemble": "#ea580c",
}


def test_order_locations(csv_path: Path) -> pd.DataFrame:
    """Recreate the environment's filtering and test split to align GPS anchors."""
    frame = pd.read_csv(csv_path)
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].astype(str).str.strip()
    frame["traffic"] = frame["Road_traffic_density"].map(TRAFFIC_MAP)
    frame["weather"] = frame["Weather_conditions"].str.split().str[-1].map(WEATHER_MAP)
    frame["multi"] = pd.to_numeric(frame["multiple_deliveries"], errors="coerce")
    time_column = next(column for column in frame if column.lower().replace(" ", "").startswith("time_taken"))
    frame["time_taken"] = pd.to_numeric(frame[time_column].astype(str).str.extract(r"(\d+)")[0])
    for column in COORDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["distance_km"] = haversine_km(
        frame[COORDS[0]], frame[COORDS[1]], frame[COORDS[2]], frame[COORDS[3]]
    )
    frame = frame.dropna(subset=["traffic", "weather", "multi", "time_taken", "distance_km"])
    frame = frame[(frame[list(COORDS)].abs() > 1).all(axis=1)]
    frame = frame[frame["distance_km"] < 50].reset_index(drop=True)
    # Matches split_orders_3way(..., seed=42): test is sampled first.
    return frame.sample(frac=0.15, random_state=42).reset_index(drop=True)


def _projection(run: dict, setup: dict, locations: pd.DataFrame):
    start_r, start_c = run["start"]
    if setup["mode"] == "dataset" and setup.get("order_indices"):
        row = locations.iloc[int(setup["order_indices"][0])]
        lat0, lon0 = float(row[COORDS[0]]), float(row[COORDS[1]])
        first_lat, first_lon = float(row[COORDS[2]]), float(row[COORDS[3]])
        first_r, first_c = run["dests"][0]
        grid_vector = complex(first_c - start_c, start_r - first_r)
        lat_scale = 111.0
        lon_scale = 111.0 * max(0.2, math.cos(math.radians(lat0)))
        geo_vector = complex((first_lon - lon0) * lon_scale, (first_lat - lat0) * lat_scale)
        rotation = (geo_vector / abs(geo_vector)) / (grid_vector / abs(grid_vector)) if (
            abs(geo_vector) > 1e-9 and abs(grid_vector) > 0
        ) else 1
        km_per_cell = max(0.35, min(1.2, float(row["distance_km"]) / max(1, abs(first_c - start_c) + abs(first_r - start_r))))
        anchor_text = "centred near the first selected dataset restaurant"
    else:
        lat0, lon0 = FALLBACK_CENTER
        rotation, km_per_cell = 1, 0.6
        anchor_text = "centred on an illustrative city location"

    lon_scale = 111.0 * max(0.2, math.cos(math.radians(lat0)))

    def project(cell):
        row, col = cell
        displacement = complex(col - start_c, start_r - row) * rotation * km_per_cell
        return [lat0 + displacement.imag / 111.0, lon0 + displacement.real / lon_scale]

    return project, anchor_text


def build_map(runs: dict, primary: str, setup: dict, locations: pd.DataFrame) -> tuple[str, str]:
    """Draw the exact simulated cell paths and stop order over real map tiles."""
    selected = runs[primary]
    project, anchor_text = _projection(selected, setup, locations)
    map_view = folium.Map(location=project(selected["start"]), zoom_start=12, tiles="OpenStreetMap", control_scale=True, scrollWheelZoom=False)
    bounds = [project(selected["start"])]

    for row, col in zip(*selected["hotspots"].nonzero()):
        point = project((int(row), int(col)))
        folium.CircleMarker(
            point, radius=4, color="#d97706", weight=1, fill=True,
            fill_color="#fbbf24", fill_opacity=0.65,
            tooltip="Simulated traffic jam",
        ).add_to(map_view)

    # Draw other policies first so the selected route remains visible on top.
    for name in [*filter(lambda item: item != primary, runs), primary]:
        points = [project(cell) for cell in runs[name]["trail"]]
        bounds.extend(points)
        folium.PolyLine(
            points, color=COLORS.get(name, "#2563eb"),
            weight=5 if name == primary else 3,
            opacity=0.9 if name == primary else 0.55,
            tooltip=name.replace("D3QN", "AI: D3QN"),
        ).add_to(map_view)

    folium.Marker(
        project(selected["start"]),
        tooltip="Rider starts here (orders already picked up)",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(map_view)
    delivered_at = {index: step for step, index in selected["events"]}
    for index, cell in enumerate(selected["dests"]):
        point = project(cell)
        bounds.append(point)
        is_new = index >= selected["n_initial"]
        colour = "#ea580c" if is_new else "#2563eb"
        deadline = selected["deadlines"][index]
        reached = delivered_at.get(index)
        tooltip = (
            f"Customer {index + 1} · target {deadline:.0f} min · "
            + (f"reached at move {reached}" if reached is not None else "not reached")
            + (" · arrived mid-trip" if is_new else "")
        )
        folium.Marker(
            point, tooltip=tooltip,
            icon=folium.DivIcon(
                html=(
                    f'<div style="background:{colour};color:white;border:2px solid white;'
                    'border-radius:50%;width:25px;height:25px;text-align:center;'
                    f'line-height:21px;font-weight:700;box-shadow:0 1px 5px #555">{index + 1}</div>'
                ),
                icon_size=(25, 25), icon_anchor=(12, 12),
            ),
        ).add_to(map_view)

    map_view.fit_bounds(bounds, padding=(25, 25))
    caption = (
        f"Real street-map background, {anchor_text}. The rider path, traffic jams and customer "
        "pins are projected from the 10×10 simulation grid. They are illustrative positions, "
        "not recorded GPS tracks or road-level directions."
    )
    return map_view.get_root().render(), caption
