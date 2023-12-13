# importing necessary packages
import os
import xarray as xr
import xroms
import numpy as np
import pandas as pd
import sys
sys.path.append('/scratch/ss4338/2020/')
import matplotlib.pyplot as plt
import matplotlib
import dask
import warnings
from track import track
from Interpolation import interpolation
from Heat_flux import HeatFlux



warnings.filterwarnings("ignore", category=FutureWarning)

#accesing data using erddap catalog
link = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/avg"
chunks = {'ocean_time':14}
ds = xroms.open_netcdf(link, chunks = chunks)


#creating a track 
bathymetry = ds.h
extent = [-76,-65, 34, 44]
sigma = 15
lon_left = -75
lon_right = -70.65
isobath_value = 65
track_coords = track(ds, bathymetry, extent, lon_left, lon_right, isobath_value, sigma)



#generating the track coordinates

AB_lon_lat, AB_ij = track_coords.AB_track(-0.74,50);
CD_lon_lat, CD_ij = track_coords.CD_track( 0.32,50);
BC_lon_lat, BC_ij = track_coords.BC_track();

final_ij_isobath = np.vstack((AB_ij[::-1], BC_ij, CD_ij))

final_lon_lat = np.vstack((AB_lon_lat[::-1], BC_lon_lat, CD_lon_lat))

lon_lat, ij_isobath = xr.DataArray(final_lon_lat), xr.DataArray(final_ij_isobath)
lon_lat, ij_isobath = lon_lat.rename({'dim_0':'along_track','dim_1':'lon_lat'}), ij_isobath.rename({'dim_0':'along_track', 'dim_1':'i_j'}) 


#creating track class instance to feed it in different class and creating interpolation instance
track_instance = track_coords 
interpol = interpolation(ds, lon_lat, ij_isobath[:,0], ij_isobath[:,1], track_instance);

# interpolating temperature and velocity
temp = interpol.temperature();
normal_vel, parallel_vel = interpol.velocity();

#saving dataset

AB_ij.to_dataset(name='AB_path').to_netcdf('./AB_path.nc')

temp.attrs.pop('grid', None)
normal_vel.attrs.pop('grid', None)

data = xr.Dataset({'temp': temp,
                 'normal_vel': normal_vel})
data.to_netcdf('./daily_data.nc')

