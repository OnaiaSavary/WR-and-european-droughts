"""
Processing of ERA5 ZG500 data:
- Linear detrending
- Smoothed daily climatology (spline)
- Anomaly computation, filtering, and normalization
- EOF analysis and weather regime clustering
"""

import xarray as xr
import numpy as np
import os
import warnings
from scipy.interpolate import splrep, BSpline
from eofs.xarray import Eof
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import pandas as pd

warnings.filterwarnings("ignore")

repository = "..."

# -----------------------------------------------------------------------------
# Load ZG500 data and select 500 hPa level
# -----------------------------------------------------------------------------
zg500 = xr.open_dataset(repository + "/zg500.nc").load().chunk(
    {"time": 1000, "latitude": 50, "longitude": 50}
)
zg500 = zg500.sel(plev="5e+04").squeeze().drop_vars("plev")

# -----------------------------------------------------------------------------
# Linear trend estimation (spatial mean)
# -----------------------------------------------------------------------------
zg500_mean = zg500.mean(dim=["longitude", "latitude"])
time_index = np.arange(len(zg500_mean.time))
coeff = np.polyfit(time_index, zg500_mean.zg500.values, 1)
trend = coeff[0] * time_index

# -----------------------------------------------------------------------------
# Remove linear trend at each grid point
# -----------------------------------------------------------------------------
print("Removing linear trend")
zg500_detrended = zg500.copy()

for i_lon in range(len(zg500.longitude)):
    for i_lat in range(len(zg500.latitude)):
        zg500_detrended["zg500"][:, i_lat, i_lon] = (
            zg500["zg500"][:, i_lat, i_lon] - trend
        )

# -----------------------------------------------------------------------------
# Daily climatology computation
# -----------------------------------------------------------------------------
print("Computing climatology")
zg500_detrended = zg500_detrended.assign_coords(
    day=zg500_detrended.time.dt.day,
    month=zg500_detrended.time.dt.month,
)

zg500_detrended.to_netcdf(repository + "/zg500_detrended.nc")
zg500_detrended = xr.open_dataset(repository + "/zg500_detrended.nc").load()

climatology = zg500_detrended.groupby("time.dayofyear").mean("time")

# -----------------------------------------------------------------------------
# Spline smoothing of the climatology
# -----------------------------------------------------------------------------
climatology_spline = climatology.copy()

for i_lon in range(len(zg500.longitude)):
    for i_lat in range(len(zg500.latitude)):
        d = climatology.isel(latitude=i_lat, longitude=i_lon, drop=True)
        x = np.arange(len(d.dayofyear) * 3)
        y = np.tile(d.zg500.values, 3)
        y[-1] = y[0] # ensure periodicity
        tck = splrep(x, y, s=len(x) * 10)
        climatology_spline["zg500"][:, i_lat, i_lon] = BSpline(*tck)(
            x[len(d.dayofyear) : 2 * len(d.dayofyear)]
        )

# -----------------------------------------------------------------------------
# Anomaly computation
# -----------------------------------------------------------------------------
print("Computing anomalies")
zg500_anomalies = (
    zg500_detrended.groupby("time.dayofyear") - climatology_spline
)
zg500_anomalies = zg500_anomalies.drop_vars(["dayofyear", "month", "day"])
zg500_anomalies.to_netcdf(repository + "/zg500_anomalies.nc")

# -----------------------------------------------------------------------------
# Temporal filtering (5-day running mean)
# -----------------------------------------------------------------------------
print("Filtering anomalies")
zg500_filtered = zg500_anomalies.rolling(time=5, center=True).mean()
zg500_filtered.to_netcdf(repository + "/zg500_filtered_anomalies.nc")

# -----------------------------------------------------------------------------
# Normalization factor (31-day rolling std, spatial mean)
# -----------------------------------------------------------------------------
print("Computing normalization factor")
zg500_std = (
    zg500_anomalies.rolling(time=31, center=True)
    .std()
    .mean(dim=("longitude", "latitude"))
)
zg500_std.to_netcdf(repository + "/zg500_std_anomalies.nc")

# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
print("Normalizing data")
zg500_filtered = xr.open_dataset(
    repository + "/zg500_filtered_anomalies.nc"
).load()
zg500_std = xr.open_dataset(
    repository + "/zg500_std_anomalies.nc"
).load()

zg500_normalized = zg500_filtered.zg500 / zg500_std.zg500
zg500_normalized.to_netcdf(
    repository + "/zg500_normalized_anomalies.nc"
)

# -----------------------------------------------------------------------------
# EOF analysis (1991–2020)
# -----------------------------------------------------------------------------
zg500 = xr.open_dataset(repository + "/zg500_normalized_anomalies.nc")
zg500_sel = zg500.sel(time=zg500.time.dt.year.isin(range(1990, 2020)))
zg500_coarse = zg500_sel.coarsen(longitude=3, latitude=3, boundary="trim").mean()

weights = np.cos(np.deg2rad(zg500_coarse.latitude))
weights /= weights.mean()
weights = weights.broadcast_like(zg500_coarse)

solver = Eof(zg500_coarse.zg500, weights=weights)
n_modes = 13

eofs = solver.eofsAsCovariance(neofs=n_modes)
pcs = solver.pcs(npcs=n_modes, pcscaling=1)

eofs.to_netcdf(repository + "/eofs_weighted.nc")
pcs.to_netcdf(repository + "/pcs_weighted.nc")

# -----------------------------------------------------------------------------
# Utility functions for weather regime post-processing
# -----------------------------------------------------------------------------
def decluster_short_events(cluster, min_duration):
    """
    Remove clusters with duration shorter than min_duration.
    """
    output = []
    count = 1

    for i in range(1, len(cluster)):
        if cluster[i] == cluster[i - 1]:
            count += 1
        else:
            output.extend(
                [cluster[i - 1]] * count if count > min_duration else [0] * count
            )
            count = 1

    output.extend(
        [cluster[-1]] * count if count > min_duration else [0] * count
    )
    return np.array(output)


def relabel_clusters(reference, target):
    """
    Relabel clusters by decreasing frequency.
    """
    from collections import Counter

    freq = Counter(v for v in reference if v != 0)
    order = [k for k, _ in freq.most_common()]
    mapping = {old: new + 1 for new, old in enumerate(order)}

    result = target.copy()
    for i in range(len(result)):
        result[i] = mapping.get(result[i], 0)
    return result

# -----------------------------------------------------------------------------
# K-means clustering in PC space (1991–2020)
# -----------------------------------------------------------------------------
pcs_array = pcs.values.T
kmeans = KMeans(n_clusters=7, n_init=500, max_iter=500)
kmeans.fit(pcs_array)

clusters_1991_2020 = kmeans.predict(pcs_array) + 1
centroids = kmeans.cluster_centers_

pd.DataFrame(centroids).to_csv(
    repository + "/Weather_regime/centroids.csv", index=False
)

# -----------------------------------------------------------------------------
# Projection and classification for 1960–2022
# -----------------------------------------------------------------------------
zg500 = xr.open_dataset(repository + "/zg500_normalized_anomalies.nc")
zg500 = zg500.sel(time=zg500.time.dt.year.isin(range(1960, 2023)))
zg500 = zg500.isel(time=slice(0, -16))
zg500 = zg500.coarsen(longitude=2, latitude=2, boundary="trim").mean()

EOF_ref = xr.open_dataset(repository + "/eofs_weighted.nc").eofs
EOF_flat = EOF_ref.stack(space=("latitude", "longitude")).values
zg500_flat = zg500.stack(space=("latitude", "longitude")).values

pcs_new = zg500_flat @ EOF_flat.T
pcs_new /= pcs_new.std(axis=0)

clusters = kmeans.predict(pcs_new) + 1
distance = cdist(centroids, pcs_new).T

clusters[distance.min(axis=1) >= 4.5] = 0
clusters = decluster_short_events(clusters, min_duration=3)
clusters = relabel_clusters(clusters, clusters)

pd.DataFrame(clusters).to_csv(
    repository + "/Weather_regime/clusters_1960_2022.csv", index=False
)

# -----------------------------------------------------------------------------
# Weather regime composites
# -----------------------------------------------------------------------------
zg500 = xr.open_dataset(repository + "/zg500_filtered_anomalies.nc")
zg500 = zg500.sel(time=zg500.time.dt.year.isin(range(1960, 2023)))
zg500 = zg500.isel(time=slice(0, -16))

regimes = [
    zg500.isel(time=clusters == r).mean("time")
    for r in range(8)
]

xr.concat(regimes, dim="regime").to_netcdf(
    repository + "/Weather_regime/regimes_1960_2022.nc"
)
