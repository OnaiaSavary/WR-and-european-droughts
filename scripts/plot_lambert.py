import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
import xclim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from collections import Counter
import seaborn
import xskillscore as xs
from metpy.units import units
import matplotlib.path as mpath
import scipy.stats as scs
import scipy as sc
import matplotlib.cm as cm
from matplotlib import rc
from sklearn.cluster import KMeans
import ssl
import flox #accelère l'opération "groupby"
ssl._create_default_https_context = ssl._create_unverified_context
import regionmask
import xclim
ar6_land = regionmask.defined_regions.ar6.land
from matplotlib.colors import BoundaryNorm
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')


from matplotlib import rc

plt.rcParams.update({
    'font.size': 15,                # Taille de la police par défaut
    'axes.labelsize': 'large',      # Taille de l'étiquette des axes
    'axes.titlesize': 'large',    # Taille du titre
    'legend.fontsize': 'large',    # Taille de la légende
    'lines.linewidth': 2,           # Largeur de la ligne par défaut
    #'figure.dpi': 300              # Résolution DPI par défaut pour grandes images
    #'figure.dpi': 100              # Résolution DPI par défaut
})

rc('text', usetex=True)   # Permet d'utiliser des formules LaTex dans tes légendes
plt.rc('font', family='serif')  # Police d'écriture LaTex

text_kws = dict(color="black", fontsize=8, bbox=dict(pad=0.2, color="w"))
line_kws = dict(linestyle='--', linewidth=1, color='black')

ar6_land[17].name = 'West \& Central-Europe'


ar6_all = regionmask.defined_regions.ar6.all
text_kws = dict(color="black", fontsize=8, bbox=dict(pad=0.2, color="w"))
line_kws = dict(linestyle='--', linewidth=1, color='black')
region_all = ar6_all[[16,17,18,19]]


mag_sech = 'mod'

cmap = plt.get_cmap('RdYlBu_r')

REGIME_NAME = ['Regime 0','Zonal Regime (ZO)', 'Greenland Blocking (GBL)','European Blocking (EuBL)','Atlantic Ridge (AR)','Mediterranean Trough (MTr)','Atlantic Trough (AT)',   'Scandinavian Blocking (ScBL)', ]
REGIME_NAME_RED = ['WR0','ZO','GBL','EuBL','AR','MTr','AT','ScBL']



repository = '...'

def plot_lambert(champ, champ_contour, bounds, ax, fig, label, cm_norm, bounds_cm, contour, grid, title, colorbar):
    # Réalise le tracé d'un champ, sur la plage spatiale définie dans bounds = [lonmin,lonmax,latmin,latmax], sur l'axe existant ax,
    # en appliquant un label, aux limites de colorbar bounds_cm, avec la possibilité de rajouter les contours et d'afficher la grille
    # spatiale.
    lonmin = bounds[0]
    lonmax = bounds[1]
    latmin = bounds[2]
    latmax = bounds[3]

    text_kws = dict(color="black", fontsize=0, bbox=dict(pad=0.2, color="w"))
    line_kws = dict(linestyle='--', linewidth=1, color='black')

    n = 50
    aoi = mpath.Path(
        list(zip(np.linspace(lonmin,lonmax, n), np.full(n, latmax))) + \
        list(zip(np.full(n, lonmax), np.linspace(latmax, latmin, n))) + \
        list(zip(np.linspace(lonmax, lonmin, n), np.full(n, latmin))) + \
        list(zip(np.full(n, lonmin), np.linspace(latmin, latmax, n)))
    )
    
    ax.set_boundary(aoi, transform=ccrs.PlateCarree())
        
    ax.set_extent((lonmin, lonmax, latmin, latmax))
    ax.add_feature(cf.LAND, zorder=1, edgecolor='k')

    if grid == True :
    
        gl = ax.gridlines(
            draw_labels=True, rotate_labels=False,
            x_inline=False, y_inline=False,
        )

    

    if cm_norm == True :
        norm = BoundaryNorm(bounds_cm, ncolors=256, extend='both')
        if colorbar == True :
            g = ax.contourf(champ.longitude, champ.latitude, champ.values, levels = bounds_cm, cmap='RdBu_r',robust=True, extend = 'both',transform=ccrs.PlateCarree())
            fig.colorbar(g, orientation='horizontal', fraction = 0.3, extendrect=True, aspect=65, location = 'bottom', shrink = 1, label = label, ticks = bounds_cm, anchor = (0.5,0),extend = 'both')
            
        else :
            g = ax.contourf(champ.longitude, champ.latitude, champ.values, levels = bounds_cm,cmap='RdBu_r',robust=True,extend = 'both',transform=ccrs.PlateCarree())

    else :
        if colorbar == True :
            g = champ.plot(x="longitude", y="latitude",cmap='RdBu_r' ,robust=True,ax = ax,
                   transform=ccrs.PlateCarree(),extend = 'both',cbar_kwargs={'orientation': 'horizontal', 'fraction' : 0.3, 'extendrect':'True', 'aspect':65, 'location' : 'bottom', 'shrink' : 1, 'label' : label, 'anchor' : (0.5,0)})
        else :
            g = champ.plot(x="longitude", y="latitude",cmap='RdBu_r' ,robust=True,ax = ax,transform=ccrs.PlateCarree(), extend = 'both', add_colorbar = False)
    
    if contour == True :
        levels = np.linspace(np.min(champ_contour),np.max(champ_contour),10)
        ax.contour(champ_contour.longitude.values,champ_contour.latitude.values,champ_contour, levels=levels,colors='white', transform=ccrs.PlateCarree(), linewidth = 2)
    
    #cluster_da = xr.open_dataset(repository + '1_WR_ERA5/Clustering/mask_clust_'+mag_sech+'1_1991_2020.nc')
    #trace_clust(cluster_da, fill_clust = False, ax=ax, label = True)

    if title != False:
        ax.set_title(title ,fontsize = 50)
        
    ax.coastlines()

    return g

def discretize_polygon(vertices, n):
    vertices_discr = []
    for i in range(len(vertices)):
        start = np.array(vertices[i])
        end = np.array(vertices[(i + 1) % len(vertices)])
        for j in range(n + 1):
            t = j / n
            point = (1 - t) * start + t * end
            vertices_discr.append(tuple(point))
    return vertices_discr
    

def trace_clust(cluster_da, fill_clust,ax, label):
    
    INT_list = list(np.unique(cluster_da.clusters)[~np.isnan(np.unique(cluster_da.clusters))])
    
    
    for i in INT_list:
        mask = (cluster_da.clusters == i)
        contour = ax.contour(
            cluster_da.lon,
            cluster_da.lat,
            cluster_da.clusters == i, 
            levels=[0.5, 1.5],
            transform=ccrs.PlateCarree(),
            colors='black',
            linestyles='-',
            linewidths=1
        )
    

    if fill_clust == True :
        cluster_da.clusters.plot(x='lon', y='lat', ax = ax, cmap='gnuplot', transform=ccrs.PlateCarree(), add_colorbar = False)

    if label == True :
        for i in range(1,len(INT_list)+1):
            bary_x = np.mean(cluster_da.lon.where(cluster_da.clusters == i))
            bary_y = np.mean(cluster_da.lat.where(cluster_da.clusters == i))
            ax.text(bary_x, bary_y, i, transform=ccrs.PlateCarree(), fontsize = 20)
    



print('Importing clusters and regimes')
mask_cluster = xr.open_dataset(repository + '1_WR_ERA5/Clustering/mask_clust.nc')
cluster_mod = pd.read_csv(repository + '1_WR_ERA5/Weather_regime/clusters_1960_2022.csv').values.T[0]
REGIMES = xr.open_dataset(repository + '1_WR_ERA5/Weather_regime/regimes_1960_2022.nc').chunk({'latitude' :50, 'longitude' :50}).load()

REGIMES_bueller = xr.open_dataset(repository + '1_WR_ERA5/Weather_regime/regimes_1960_2022_bueller.nc')
zg500_ctr2 = xr.open_dataset(repository + 'ERA5/zg500/zg500_detrend_1960_2023.nc').load().chunk({'time':1000,'latitude' : 20, 'longitude' : 20}).isel(time= slice(0,-16))

lonmin=-75
lonmax=64
latmin=28
latmax=75

bounds = [lonmin,lonmax,latmin,latmax]
born = 100

bounds_cm = list(np.linspace(-born,born,11))

proj = ccrs.LambertConformal(central_longitude=(lonmin+lonmax)/2, central_latitude=(latmin+latmax)/2)
fig, axe = plt.subplots(2,4, figsize = (40,25), facecolor="w",subplot_kw=dict(projection=proj),dpi = 500)

zg500_geop = zg500_ctr2.drop_vars(('time_bnds', 'day', 'month'))

for regime in range(1,8):
    print(regime)
    print(REGIME_NAME[regime])
    ax = axe[(regime-1)//4,(regime-1)%4]
    freq_reg = len(cluster_mod[cluster_mod==regime])/len(cluster_mod)
    print(freq_reg)
    ZG_clust = REGIMES_bueller.sel({'regime':regime})
    
    zg500_contour = zg500_geop.isel({'time': cluster_mod == regime}).mean(dim = 'time')
    
    g = plot_lambert(ZG_clust.zg500, zg500_contour.zg500, bounds, ax, fig, label = '', cm_norm = True, bounds_cm = bounds_cm, contour = True, grid = False, title = REGIME_NAME[regime]+'\n'+str(int(freq_reg*1000)/10)+'\%', colorbar = False)
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree())
    ax.coastlines()

for regime in range(0,1):
    ax = axe[1,3]
    freq_reg = len(cluster_mod[cluster_mod==regime])/len(cluster_mod)
    ZG_clust = REGIMES_bueller.sel({'regime':regime})
    
    zg500_contour = zg500_geop.isel({'time': cluster_mod == regime}).mean(dim = 'time')
    
    g = plot_lambert(ZG_clust.zg500, zg500_contour.zg500, bounds, ax, fig, label = '', cm_norm = True, bounds_cm = bounds_cm, contour = True, grid = False, title = 'Regime 0 \n'+str(int(freq_reg*1000)/10)+'\%', colorbar = False)
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree())
    ax.coastlines()

fig.tight_layout()
cbar = fig.colorbar(g,cax = None, ax = axe[:,:],  extendrect=True, aspect=65, location = 'bottom', shrink = 1, ticks = bounds_cm)
cbar.ax.tick_params(labelsize=50)
cbar.set_label(label = 'Geopotential height anomaly [gpm]', size=50, weight='bold')

plt.savefig(repository + 'figures/WR.svg')

plt.show()


