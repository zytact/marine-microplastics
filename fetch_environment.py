"""Downloads the environmental datasets used in section 5.1 of microplastics.py into data/environment.

Usage: uv run python fetch_environment.py

Files that already exist are kept. The drifter climatology is too large to keep at 0.25 degrees, so only
its 1 degree average is saved. Sources and licences are listed in README.md.
"""

from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

OUT = Path("data/environment")
RAW = {
    "ne_10m_coastline.zip": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip",
    "ne_10m_populated_places_simple.zip": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip",
    "Meijer2021_midpoint_emissions.zip": "https://ndownloader.figshare.com/files/27807774",
    "sst.ltm.1991-2020.nc": "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.ltm.1991-2020.nc",
}
DRIFTER = "https://erddap.aoml.noaa.gov/gdp/erddap/griddap/drifter_annualmeans.csv?"
DRIFTER_CSV = "drifter_currents_1deg.csv"


def drifter_currents() -> pd.DataFrame:
    """Annual mean near-surface velocity from the Global Drifter Program, averaged from 0.25 to 1 degree cells.
    ERDDAP times out on the full grid, so it is requested in 10 degree latitude bands."""
    bands = []
    for south in np.arange(-72.875, 84.875, 10):
        north = min(south + 9.75, 84.875)
        box = f"[(-179.875):1:(179.875)][({south}):1:({north})]"
        bands.append(pd.read_csv(DRIFTER + quote(f"U{box},V{box}", safe="(),:"), skiprows=[1]))
    cells = pd.concat(bands).dropna()
    cells["lat"] = np.floor(cells.latitude) + 0.5
    cells["lon"] = np.floor(cells.longitude) + 0.5
    return cells.groupby(["lat", "lon"]).agg(u=("U", "mean"), v=("V", "mean")).round(4).reset_index()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in RAW.items():
        if not (OUT / name).exists():
            urlretrieve(url, OUT / name)
    if not (OUT / DRIFTER_CSV).exists():
        drifter_currents().to_csv(OUT / DRIFTER_CSV, index=False)
    for path in sorted(OUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
