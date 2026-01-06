# WR-and-european-droughts

Directory corresponding to the article ‘Linking European droughts to year-round weather regimes’ by Savary et al. (https://doi.org/10.5194/egusphere-2025-3308).

The “scripts” folder contains the scripts used to obtain the main results.
The data and the Jupyter notebook for generating figures, which are too large to be uploaded to GitHub, are available on Zenodo at ...


 - **SPI_daily.py** code for calculating the Standardised Precipitation Index with a 3-month integration period, using the pr_land.nc dataset for the period 1960-2022

 - **weather_regimes.py** code for calculating all-season weather regimes from the zg500.nc dataset, following the procedure described by Grams et al. (https://doi.org/10.1038/nclimate3338) and adapted to our study

 - **plot_lambert.py** turnkey plotting code that can be used to plot a spatial field in Lambert projection. This plot is used in our study to plot weather regimes.
