import xarray as xr
import numpy as np

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
