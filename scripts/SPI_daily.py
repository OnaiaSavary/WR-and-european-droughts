"""
ERA5 precipitation processing:
- Spatial masking (land and AR6 regions)
- Extraction of clustered weather regime periods
- Daily SPI (Standardized Precipitation Index) computation
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr

import xclim
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import metpy.calc as mpcalc
from metpy.units import units
from matplotlib import rc
import ssl
import regionmask
from matplotlib.colors import BoundaryNorm

ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Plot configuration
# -----------------------------------------------------------------------------
rc("text", usetex=True)
plt.rc("font", family="serif")

text_kws = dict(color="black", fontsize=8, bbox=dict(pad=0.2, color="w"))
line_kws = dict(linestyle="--", linewidth=1, color="black")

# -----------------------------------------------------------------------------
# AR6 land regions definition
# -----------------------------------------------------------------------------
ar6_land = regionmask.defined_regions.ar6.land
ar6_land[17].name = "West & Central Europe"

ar6_all = regionmask.defined_regions.ar6.all
region_all = ar6_all[[16, 17, 18, 19]]
Region = [16, 17, 18, 19]

# -----------------------------------------------------------------------------
# Load precipitation data
# -----------------------------------------------------------------------------
print("Loading precipitation data")
pr = xr.open_dataset(repository + "ERA5/pr/pr.nc")

# Rename coordinates if needed
try:
    pr = pr.rename({"latitude": "lat", "longitude": "lon"})
except Exception:
    pass

pr = pr.sortby(pr.lat)
pr["time"] = pr.indexes["time"].normalize()
pr["pr"] = pr["pr"].assign_attrs({"units": "mm day-1"})

# -----------------------------------------------------------------------------
# Apply weather regime cluster mask
# -----------------------------------------------------------------------------
mask_cluster = xr.open_dataset(
    repository + "Clustering/mask_clust_" + mag_sech + "1_1991_2020.nc"
)

pr_reg = pr.where(
    mask_cluster.clusters.isin([1, 2, 3, 4, 5, 6, 7]),
    drop=True,
).isel(time=slice(0, -16))

# -----------------------------------------------------------------------------
# Remove ocean grid points (land-only mask)
# -----------------------------------------------------------------------------
land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110
land_mask = land_110.mask_3D(pr)
mask_lsm = land_mask.squeeze(drop=True)
pr_reg = pr.where(mask_lsm)

# -----------------------------------------------------------------------------
# Restrict domain to selected AR6 land regions
# -----------------------------------------------------------------------------
mask_3D = ar6_land.mask_3D(pr_reg)
combined_mask = sum(mask_3D.sel(region=i) for i in [16, 17, 18, 19])
pr_reg = pr_reg.where(combined_mask)

# -----------------------------------------------------------------------------
# Save land-only precipitation dataset
# -----------------------------------------------------------------------------
print("Exporting precipitation dataset")
pr_reg.to_netcdf(
    repository + "ERA5/pr/pr_land.nc",
    encoding={
        "time": {"dtype": "double"},
        "pr": {"dtype": "double"},
        "lon": {"dtype": "double"},
        "lat": {"dtype": "double"},
    },
)

# -----------------------------------------------------------------------------
# SPI computation
# -----------------------------------------------------------------------------
repository = "..."

from standard_precip import spi
from standard_precip.utils import plot_index

pr = xr.open_dataset(repository + "ERA5/pr/pr_land.nc")
pr = pr.sel(time=pr.time.dt.year.isin(np.arange(1960, 2023)))

# Initialize output array
SPI_calc = xr.full_like(pr, np.nan, dtype=np.double)

scale_days = 90 # SPI accumulation window (days)

# -----------------------------------------------------------------------------
# Grid-point SPI calculation
# -----------------------------------------------------------------------------
for lon_ind in range(len(pr.lon)):
    print(lon_ind / len(pr.lon) * 100, end="\r")
    for lat_ind in range(len(pr.lat)):

        pr_grid = pr.isel(lon=lon_ind, lat=lat_ind).pr.to_dataframe()

        # Skip grid points with missing data
        if not np.isnan(
            pr.isel(lon=lon_ind, lat=lat_ind)
            .pr.mean(dim="time", skipna=True)
        ):

            spi_model = spi.SPI()
            df_spi = spi_model.calculate(
                pr_grid.reset_index(),
                time_col="time",
                precip_col="pr",
                freq="D",
                scale=scale_days,
                fit_type="lmom",
                dist_type="gam",
            )

            SPI_calc["pr"][:, lat_ind, lon_ind] = (
                df_spi.pr_scale_90_calculated_index.values
            )

# -----------------------------------------------------------------------------
# Save SPI dataset
# -----------------------------------------------------------------------------
SPI_calc.to_netcdf(
    repository + "ERA5/SPI/SPI_daily_1960_2022.nc",
    encoding={"time": {"dtype": "double"}},
)
