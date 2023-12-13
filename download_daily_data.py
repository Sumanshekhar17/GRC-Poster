import sys
sys.path.append('/scratch/ss4338/2020/')
import xarray as xr
import numpy as np
import pandas as pd
import dask
import xroms
import matplotlib.pyplot as plt

from track import track
from Interpolation import interpolation

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

link = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/avg"
chunk = {"ocean_time":15}
ds = xroms.open_netcdf(link, chunks=chunk)

# bathymetry 

bathymetry = ds.h

extent = [-76,-65, 34, 44]
sigma = 8
lon_left = -75
lon_right = -70.65
isobath_value = 65
track_coords = track(ds, bathymetry, extent, lon_left, lon_right, isobath_value, sigma)

# track slices

AB_lon_lat, AB_ij = track_coords.AB_track(-0.74,50);
CD_lon_lat, CD_ij = track_coords.CD_track( 0.32,50);
BC_lon_lat, BC_ij = track_coords.BC_track();

#binding all slices of the track

final_ij_isobath = np.vstack((AB_ij[::-1], BC_ij, CD_ij))

final_lon_lat = np.vstack((AB_lon_lat[::-1], BC_lon_lat, CD_lon_lat))

lon_lat, ij_isobath = xr.DataArray(final_lon_lat), xr.DataArray(final_ij_isobath)
lon_lat, ij_isobath = lon_lat.rename({'dim_0':'along_track','dim_1':'lon_lat'}), ij_isobath.rename({'dim_0':'along_track', 'dim_1':'i_j'}) 


track_instance = track_coords 
interpol = interpolation(ds, lon_lat, ij_isobath[:,0], ij_isobath[:,1], track_instance);

# interpolating temperature and velocity
temp = interpol.temperature();
nvel, pvel = interpol.velocity();
dz = interpol.dz() 
dS = xr.DataArray(interpol.delta_s(lon_lat), dims=['along_track'])

temp.attrs.pop('grid', None)
nvel.attrs.pop('grid',None)
dz.attrs.pop('grid', None)

temp.z_rho.isel(ocean_time=0).to_dataset(name='depth').to_netcdf('./depth.nc')

data = xr.Dataset({'temp':temp,
                   'nvel':nvel,
                   'depth':dz,
                   'track_resolution':dS}
                   )
data.to_netcdf('./daily_data_new.nc')