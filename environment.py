from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import shapefile
from sklearn.neighbors import BallTree

ENV_DATA = Path("data/environment")
EARTH_KM = 6371.0
ENVIRONMENT_FEATURES = ["coast_km", "river_mouth_km", "river_input_200km", "population_200km", "sst_climatology", "current_speed"]


def great_circle_tree(points: pd.DataFrame) -> BallTree:
    return BallTree(np.radians(points[["lat", "lon"]].to_numpy()), metric="haversine")


def coast_points(step_km: float = 2.0) -> pd.DataFrame:
    points = []
    for shape in shapefile.Reader(ENV_DATA / "ne_10m_coastline.zip").shapes():
        for line in np.split(np.array(shape.points), shape.parts[1:]):
            lon, lat = np.radians(line).T
            gap = 2 * EARTH_KM * np.arcsin(np.sqrt(
                np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
            ))
            steps = np.maximum(np.ceil(gap / step_km), 1).astype(int)
            segment = np.repeat(np.arange(len(gap)), steps)
            fraction = (np.arange(steps.sum()) - np.repeat(np.cumsum(steps) - steps, steps)) / np.repeat(steps, steps)
            points += [line[segment] + (line[segment + 1] - line[segment]) * fraction[:, None], line[-1:]]
    return pd.DataFrame(np.concatenate(points), columns=["lon", "lat"])


def shapefile_points(path: Path) -> pd.DataFrame:
    reader = shapefile.Reader(path)
    return pd.DataFrame([
        record.as_dict() | {"lon": shape.points[0][0], "lat": shape.points[0][1]}
        for record, shape in zip(reader.records(), reader.shapes())
    ])


def nearest_km(points: pd.DataFrame, at: pd.DataFrame) -> np.ndarray:
    return great_circle_tree(points).query(np.radians(at[["lat", "lon"]].to_numpy()))[0][:, 0] * EARTH_KM


def sum_within(points: pd.DataFrame, weight: str, at: pd.DataFrame, radius_km: float) -> np.ndarray:
    within = great_circle_tree(points).query_radius(np.radians(at[["lat", "lon"]].to_numpy()), radius_km / EARTH_KM)
    values = points[weight].to_numpy()
    return np.array([values[i].sum() for i in within])


def environmental_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    outlets = shapefile_points(ENV_DATA / "Meijer2021_midpoint_emissions.zip").rename(columns={"dots_exten": "tonnes"})
    places = shapefile_points(ENV_DATA / "ne_10m_populated_places_simple.zip")
    currents = pd.read_csv(ENV_DATA / "drifter_currents_1deg.csv").assign(speed=lambda c: np.hypot(c.u, c.v))
    with h5py.File(ENV_DATA / "sst.ltm.1991-2020.nc") as f:
        sst = f["sst"][:]
    current_cell = great_circle_tree(currents).query(np.radians(frame[["lat", "lon"]].to_numpy()))[1][:, 0]
    return pd.DataFrame({
        "coast_km": nearest_km(coast_points(), frame),
        "river_mouth_km": nearest_km(outlets.nlargest(1000, "tonnes"), frame),
        "river_input_200km": sum_within(outlets, "tonnes", frame, 200),
        "population_200km": sum_within(places, "pop_max", frame, 200),
        "sst_climatology": sst[
            frame.date.dt.month - 1,
            np.clip(np.floor(90 - frame.lat), 0, 179).astype(int),
            np.floor(frame.lon % 360).astype(int) % 360,
        ],
        "current_speed": currents.speed.to_numpy()[current_cell],
    }, index=frame.index)
