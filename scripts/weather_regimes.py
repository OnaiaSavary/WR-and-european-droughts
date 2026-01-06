import xarray as xr
import numpy as np



import os
import sys
import glob
import re
import numpy as np
import xarray as xr
import flox #accelère l'opération "groupby"
import xclim
import matplotlib.pyplot as plt
from metpy.cbook import get_test_data
from metpy.units import units
from matplotlib import rc
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import xclim
from scipy.interpolate import splrep, BSpline

import warnings
warnings.filterwarnings('ignore')


repository = '...'

zg500 = xr.open_dataset(repository+'/zg500.nc').load().chunk({'time':1000,'latitude' : 50, 'longitude' : 50})
zg500 = zg500.sel({'plev':'5e+04'}).squeeze().drop_vars(['plev'])

# Evaluation de la tendance linéaire
zg500_mean = zg500.mean(dim=['longitude', 'latitude'])

long = len(zg500_mean.time.values)
x = np.arange(0,long)
y = zg500_mean.zg500.values
coeff = np.polyfit(x, y, 1)
zg500_detrend = zg500.copy()
trend = coeff[0]*x

# Retrait de la tendance linéaire
print('Retrait de la tendance linéaire')
for lon_ind in range (0,len(zg500.longitude.values)) :
    for lat_ind in range (0,len(zg500.latitude.values)) :
        zg500_detrend['zg500'][:,lat_ind,lon_ind] = zg500['zg500'][:,lat_ind,lon_ind] - trend



# Calcul de la climatologie
print('Calcul de la climatologie')
zg500_detrend = zg500_detrend.assign_coords(day=zg500_detrend['time.day'], month=zg500_detrend['time.month']).compute()

zg500_detrend.to_netcdf(repository+'/zg500_detrend.nc', encoding = {"time": {"dtype": "double"}})
zg500_detrend = xr.open_dataset(repository+'/zg500_detrend.nc').load().chunk({'time':1000,'latitude' : 50, 'longitude' : 50})
climatology = zg500_detrend.groupby('time.dayofyear')
climatology = climatology.mean(dim='time').compute()

# Lissage spline de la climatologie
climatology_spline = climatology.copy()
for lon_ind in range (0,len(zg500_detrend.longitude.values)) :
    for lat_ind in range (0,len(zg500_detrend.latitude.values)) :
        # Lissage spline de la climatologie
        d = climatology.isel({'latitude':lat_ind, 'longitude':lon_ind}, drop = True)
        x = np.arange(0,len(d.dayofyear.values)*3)
        y = np.tile(d['zg500'].values,3)
        y[len(y)-1] = y[0]
        tck_s = splrep(x, y, s=len(x)*10)
        climatology_spline['zg500'][:,lat_ind,lon_ind] = BSpline(*tck_s)(x)[len(d.dayofyear.values):len(d.dayofyear.values)*2]

#Calcul des anomalies
print('Calcul des anomalies')
zg500_anomalies = zg500_detrend.drop_vars('time_bnds').groupby("time.dayofyear") - climatology_spline
zg500_anomalies = zg500_anomalies.drop_vars(['dayofyear','month','day'])
zg500_anomalies.to_netcdf(repository+'/zg500_anomalies.nc', encoding = {"time": {"dtype": "double"}})



#Filtrage
print('Filtrage')
zg500_filtered = zg500_anomalies.rolling(time=5, center=True).mean()
zg500_filtered.to_netcdf(repository+'/zg500_filtered_anomalies.nc', encoding = {"time": {"dtype": "double"}})
        

# Calcul du coeff de normalisation, par moyenne spatiale de l'écart type glissant sur 31 jours
print('Calcul du coeff de normalisation')
zg500_std = zg500_anomalies.rolling(time=31, center=True).std().mean(dim = ('longitude','latitude'))
zg500_std.to_netcdf(repository+'/zg500_std_anomalies.nc', encoding = {"time": {"dtype": "double"}})

# Normalisation des données
print('Normalisation')
zg500_std = xr.open_dataset(repository+'/zg500_std_anomalies.nc').load()
zg500_filtered = xr.open_dataset(repository+'/zg500_filtered_anomalies.nc').load()

zg500_normalized = zg500_filtered['zg500'][:]/zg500_std.zg500

zg500_normalized.to_netcdf(repository+'/zg500_normalized_anomalies.nc', encoding = {"time": {"dtype": "double"}})

import os
os.remove(repository+'/zg500_detrend_day.nc')
os.remove(repository+'/climatology_extended.nc')
os.remove(repository+'/zg500_anomalies.nc')
os.remove(repository+'/zg500_std_anomalies.nc')


import xarray as xr
import numpy as np
from eofs.xarray import Eof 

zg500 = xr.open_dataset(repository+'/zg500_normalized_anomalies.nc')

zg500_notnan = zg500.sel(time = zg500['time'].dt.year.isin(np.arange(1990,2020)))
zg500_coarse = zg500_notnan.coarsen(longitude=3, latitude=3, boundary='trim').mean()
weights = np.cos(np.deg2rad(zg500_coarse.latitude))
weights /= weights.mean()
w = weights.broadcast_like(zg500_coarse)

solver = Eof(zg500_coarse.zg500, weights = w)

print('Solveur : ')
print(solver)

n=13

eofs = solver.eofsAsCovariance(neofs=n).compute()
print('EOFS : ')
print(eofs)
eofs.to_netcdf(repository+'/eofs_weighted.nc')
pcs = solver.pcs(npcs=n, pcscaling=1).compute()
print('PCS : ')
print(pcs)
pcs.to_netcdf(repository+'/pcs_weighted.nc')



EOF_1991_2020 = xr.open_dataset(repository + '1_WR_ERA5/EOF/eofs_1991_2020.nc')
pcs_1991_2020 = xr.open_dataset(repository + '1_WR_ERA5/EOF/pcs_1991_2020.nc')

n_clust = 7
n_composante = 13

repository = '...'

def declassification(cluster,jours_min):
    result = []
    count = 1
    
    # Parcourir la liste pour détecter les séquences
    for i in range(1, len(cluster)):
        if cluster[i] == cluster[i - 1]:
            count += 1
        else:
            if count <= jours_min:
                result.extend([0] * count)
            else:
                result.extend([cluster[i - 1]] * count)
            count = 1 # Reset le compteur

    # Gérer la dernière séquence
    if count <= jours_min:
        result.extend([0] * count)
    else:
        result.extend([cluster[-1]] * count)

    return result





def renumeroter_clusters(a,b) :
    
    L1 = dict(Counter(val for val in a if val != 0))
    L2 = dict(sorted(L1.items(), key=lambda x: (-x[1], x[0])))
    print(L1)
    print(L2)
    dico_corr = {}
    for i in range(0,len(list(L2))) :
        dico_corr[list(L2)[i]] = sorted(np.unique([k for k in a if k!=0]))[i]
    print(dico_corr)
    L_fin_a = a.copy()
    L_fin_b = b.copy()
    for i in range(0,len(a)):
        if a[i]!=0 :
            L_fin_a[i] = dico_corr[a[i]]
        if b[i]!=0 :
            L_fin_b[i] = dico_corr[b[i]]
        else :
            L_fin_a[i] = 0
            L_fin_b[i] = 0

    return L_fin_b



n_clust = 7
n_composante = 13

n_init=500
max_iter=500


PC_1991_2020 = []
for mod in range(0,n_composante+1):
    PC_1991_2020.append(pcs_1991_2020.sel({'mode':mod}).pcs)
pcs_T_1991_2020=np.array(PC_1991_2020).transpose()
print(pcs_T_1991_2020)

kmeans = KMeans(n_clusters=n_clust, n_init=n_init, max_iter=max_iter, verbose=0)
kmeans = kmeans.fit(pcs_T_1991_2020)


# Réattribution sur 1991-2020
# Getting the cluster labels
cluster_1991_2020 = kmeans.predict(pcs_T_1991_2020)
cluster_1991_2020 = cluster_1991_2020+1
# Centroid values
centroids = kmeans.cluster_centers_
pd.DataFrame(centroids).to_csv(repository + '1_WR_ERA5/Weather_regime/temp_centroids.csv', index=False)

from scipy.spatial.distance import cdist
distance = cdist(centroids, pcs_T_1991_2020).T

cluster_mod_1991_2020_bis = np.array(declassification(cluster_1991_2020,jours_min=3))
cluster_mod_1991_2020_bis[[i for i in range(0,len(distance)) if (distance[i]>=4.5).all()]] = 0
cluster_mod_1991_2020 = np.array(renumeroter_clusters(cluster_mod_1991_2020_bis, cluster_mod_1991_2020_bis))

pd.DataFrame(cluster_mod_1991_2020).to_csv(repository + '1_WR_ERA5/Weather_regime/clusters_1991_2020.csv', index=False)
#pd.DataFrame(cluster_mod_1991_2020).to_csv(repository + 'ERA5/psl/clusters_1991_2020_SLP.csv', index=False)

new_centroids = np.zeros(np.shape(centroids))
for i in range(0,7):
    new_centroids[i,:] = np.mean(pcs_T_1991_2020[cluster_mod_1991_2020 == i], axis = 0)





zg500 = xr.open_dataset(repository + 'ERA5/zg500/zg500_normalized_anomalies.nc').chunk({'time':1000,'latitude' : 20, 'longitude' : 20}).load()
zg500_notnan = zg500.sel(time = zg500['time'].dt.year.isin(np.arange(1960,2023))).isel(time = slice(0,-16)).zg500
zg500_notnan = zg500_notnan.coarsen(longitude=2, latitude=2, boundary='trim').mean()

EOF_1991_2020 = xr.open_dataset(repository + '1_WR_ERA5/EOF/eofs_1991_2020.nc').eofs
EOF_1991_2020 = EOF_1991_2020.sel(mode = EOF_1991_2020['mode'].isin(np.arange(0,14)), drop = True)

EOF_1991_2020_st = EOF_1991_2020.stack(space=("latitude", "longitude")) # (mode, points)
EOF_1991_2020_np = EOF_1991_2020_st.values # numpy array


zg500_notnan_flat = zg500_notnan.stack(space=("latitude", "longitude")) # (time, points)
zg500_notnan_flat_np = zg500_notnan_flat.values # numpy array

pcs_new = np.dot(zg500_notnan_flat_np, EOF_1991_2020_np.T) # (time, mode)

pcs_new = pcs_new/pcs_new.std(axis = 0) 

pcs_da = xr.DataArray(pcs_new,
                      dims=("time", "mode"),
                      coords={"time": zg500_notnan.time, "mode": EOF_1991_2020.mode})


PC_1960_2023 = []
for mod in range(0,n_composante+1):
    PC_1960_2023.append(pcs_da.sel({'mode':mod}))

pcs_T_1960_2023=np.array(PC_1960_2023).transpose()

cluster_1960_2023 = kmeans.predict(pcs_T_1960_2023)
cluster_1960_2023 = cluster_1960_2023+1

distance = cdist(centroids, pcs_T_1960_2023).T

DA_correlation = [np.argmin(distance[i])+1 for i in range(0, len(distance))]

DA_correlation = np.array(DA_correlation)
distance = cdist(centroids, pcs_T_1960_2023).T
DA_correlation[[i for i in range(0,len(distance)) if (distance[i]>=4.5).all()]] = 0
cluster_mod_1960_2022_bis = np.array(declassification(DA_correlation,jours_min=3))

cluster_mod_1960_2022 = np.array(renumeroter_clusters(cluster_mod_1960_2022_bis ,cluster_mod_1960_2022_bis ))
pd.DataFrame(cluster_mod_1960_2022).to_csv(repository + '1_WR_ERA5/Weather_regime/clusters_1960_2022.csv', index=False)



zg500 = xr.open_dataset(repository + 'ERA5/zg500/zg500_normalized_anomalies.nc')
zg500_notnan = zg500.sel(time = zg500['time'].dt.year.isin(np.arange(1960,2023))).isel(time = slice(0,-16))
R = []
for reg in range(0,n_clust+1):
    R.append(zg500_notnan.isel({'time': cluster_mod_1960_2022 == reg}))
REGIMES = xr.concat(R, dim = 'regime').mean('time')
REGIMES.to_netcdf(repository + '1_WR_ERA5/Weather_regime/regimes_1960_2022.nc')



cluster_mod_1960_2022 = pd.read_csv(repository + '1_WR_ERA5/Weather_regime/clusters_1960_2022.csv').values.T[0]
zg500 = xr.open_dataset(repository + 'ERA5/zg500/zg500_filtered_anomalies.nc')
zg500_notnan = zg500.sel(time = zg500['time'].dt.year.isin(np.arange(1960,2023))).isel(time = slice(0,-16))
R = []
for reg in range(0,n_clust+1):
    R.append(zg500_notnan.isel({'time': cluster_mod_1960_2022 == reg}))
REGIMES_bueller = xr.concat([R[i].mean(dim = 'time') for i in range(0,len(R))], dim = 'regime')
REGIMES_bueller.to_netcdf(repository + '1_WR_ERA5/Weather_regime/regimes_1960_2022_bueller.nc')
