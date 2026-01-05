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
