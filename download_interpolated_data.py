import sys
sys.path.append('/scratch/ss4338/2020/')
from track import track
from Interpolation import interpolation
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import xroms

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# accessing dataset
link = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/his"
chunk = {"ocean_time":50}
ds = xroms.open_netcdf(link, chunks=chunk)

start_time = '2020-01-01'
end_time = '2021-01-01'

ds = (ds).sel(ocean_time=slice(start_time, end_time))


bathymetry = ds.h

extent = [-76,-65, 34, 44]
sigma = 15
lon_left = -75
lon_right = -70.65
isobath_value = 65
track_coords = track(ds, bathymetry, extent, lon_left, lon_right, isobath_value, sigma)








#generating the track coordinates
AB_lon_lat, AB_ij = track_coords.AB_track(-0.74,50);

#creating track class instance to feed it in different class and creating interpolation instance
track_instance = track_coords 
interpol = interpolation(ds, AB_lon_lat[::-1], AB_ij[:,0][::-1], AB_ij[:,1][::-1], track_instance);

# interpolating temperature and velocity
AB_temp = interpol.temperature();
AB_normal_vel, AB_parallel_vel = interpol.velocity();






#generating the track coordinates
BC_lon_lat, BC_ij = track_coords.BC_track();

#creating track class instance to feed it in different class and creating interpolation instance
track_instance = track_coords 
interpol = interpolation(ds, BC_lon_lat, BC_ij[:,0], BC_ij[:,1], track_instance);

# interpolating temperature and velocity
BC_temp = interpol.temperature();
BC_normal_vel, BC_parallel_vel = interpol.velocity();




#generating the track coordinates
CD_lon_lat, CD_ij = track_coords.CD_track( 0.32,50);

#creating track class instance to feed it in different class and creating interpolation instance
track_instance = track_coords 
interpol = interpolation(ds, CD_lon_lat, CD_ij[:,0], CD_ij[:,1], track_instance);

# interpolating temperature and velocity
CD_temp = interpol.temperature();
CD_normal_vel, CD_parallel_vel = interpol.velocity();



AB_temp.attrs.pop('grid', None)
AB_temp.z_rho.chunk({'ocean_time': 50}).to_dataset(name='AB_depth').to_netcdf('./AB_depth.nc')
AB_temp.chunk({'ocean_time': 50}).to_dataset(name='AB_temp').to_netcdf('./AB_temp_2020.nc')

AB_normal_vel.attrs.pop('grid', None)
AB_normal_vel.chunk({'ocean_time': 50}).to_dataset(name='AB_normal_vel').to_netcdf('./AB_nvel_2020.nc')






BC_normal_vel.attrs.pop('grid', None)
BC_normal_vel.chunk({'ocean_time': 50}).to_dataset(name='BC_normal_vel').to_netcdf('./BC_nvel_2020.nc')

BC_temp.attrs.pop('grid', None)
BC_temp.chunk({'ocean_time': 50}).to_dataset(name='BC_temp').to_netcdf('./BC_temp_2020.nc')




CD_normal_vel.attrs.pop('grid', None)
CD_normal_vel.chunk({'ocean_time': 50}).to_dataset(name='CD_normal_vel').to_netcdf('./CD_nvel_2020.nc')

CD_temp.attrs.pop('grid', None)
CD_temp.chunk({'ocean_time': 50}).to_dataset(name='CD_temp').to_netcdf('./CD_temp_2020.nc')
