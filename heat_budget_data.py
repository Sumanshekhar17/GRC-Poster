import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import xroms
import dask
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from Heat_flux import HeatFlux
from track import track

current_directory = os.getcwd()
year = current_directory.split(os.path.sep)[-1] 

# accessing dataset
link = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/his"
chunk = {"ocean_time":50}
ds = xroms.open_netcdf(link, chunks=chunk)

start = int(year)
end = start + 1
start_time = f'{start}-01-01'
end_time = f'{end}-01-01'

ds = (ds).sel(ocean_time=slice(start_time, end_time))


bathymetry = ds.h

extent = [-76,-65, 34, 44]
sigma = 15
lon_left = -75
lon_right = -70.65
isobath_value = 65

#defining track instances
track_coords = track(ds, bathymetry, extent, lon_left, lon_right, isobath_value, sigma)

rho, cp = 1027.8, 3850


#generating the track coordinates
AB_lon_lat, AB_ij = track_coords.AB_track(-0.74,50);

heat_flux = HeatFlux(ds, AB_lon_lat[::-1], AB_ij[:,0][::-1], AB_ij[:,1][::-1], rho, cp)
temp = xr.open_dataset(f'./AB_temp_{year}.nc')
vel = xr.open_dataset(f'./AB_nvel_{year}.nc')

vel = vel.AB_normal_vel
temp = temp.AB_temp

Eddy_AB = heat_flux.Eddy_variability(temp,vel, track_coords)









BC_lon_lat, BC_ij = track_coords.BC_track();

heat_flux = HeatFlux(ds, BC_lon_lat, BC_ij[:,0], BC_ij[:,1], rho, cp)
temp = xr.open_dataset(f'./BC_temp_{year}.nc')
vel = xr.open_dataset(f'./BC_nvel_{year}.nc')

vel = vel.BC_normal_vel
temp = temp.BC_temp

Eddy_BC = heat_flux.Eddy_variability(temp,vel, track_coords)








CD_lon_lat, CD_ij = track_coords.CD_track( 0.32,50);

heat_flux = HeatFlux(ds, CD_lon_lat, CD_ij[:,0], CD_ij[:,1], rho, cp)
temp = xr.open_dataset(f'./CD_temp_{year}.nc')
vel = xr.open_dataset(f'./CD_nvel_{year}.nc')

vel = vel.CD_normal_vel
temp = temp.CD_temp

Eddy_CD = heat_flux.Eddy_variability(temp,vel, track_coords)

# Save the values
xr.DataArray(Eddy_AB, name='Eddy_AB').to_netcdf(f'Eddy_AB_{year}.nc')
xr.DataArray(Eddy_BC, name='Eddy_BC').to_netcdf(f'Eddy_BC_{year}.nc')
xr.DataArray(Eddy_CD, name='Eddy_CD').to_netcdf(f'Eddy_CD_{year}.nc')

