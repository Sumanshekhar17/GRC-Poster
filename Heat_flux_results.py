import sys
sys.path.append('/scratch/ss4338/2020/')
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




def gather_data(prefix, variable, years):
    data_list = []            
    for year in years:
        ds = xr.open_dataset(f'./{year}/{prefix}_{variable}_{year}.nc', chunks={'ocean_time': 10})
        
        if variable == 'nvel':
            data = ds[f'{prefix}_normal_vel']
        else:
            data = ds[f'{prefix}_temp']

        # Drop duplicates in 'ocean_time' dimension

        data = data.isel(ocean_time=~data.ocean_time.to_series().duplicated())

        data_list.append(data)
    return xr.concat(data_list, dim="ocean_time")

years = np.arange(2007, 2021)

AB_vel_data = gather_data('AB', 'nvel', years)
BC_vel_data = gather_data('BC', 'nvel', years)
CD_vel_data = gather_data('CD', 'nvel', years)

AB_temp_data = gather_data('AB', 'temp', years)
BC_temp_data = gather_data('BC', 'temp', years)
CD_temp_data = gather_data('CD', 'temp', years)




# accessing dataset
link = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/his"
chunk = {"ocean_time":10}
ds = xroms.open_netcdf(link, chunks=chunk)

bathymetry = ds.h

extent = [-76,-65, 34, 44]
sigma = 15
lon_left = -75
lon_right = -70.65
isobath_value = 65
track_coords = track(ds, bathymetry, extent, lon_left, lon_right, isobath_value, sigma)

rho, cp = 1027.8, 3850


#AB heat flux
AB_lon_lat, AB_ij = track_coords.AB_track(-0.74,50);

ABheat_flux = HeatFlux(ds, AB_lon_lat[::-1], AB_ij[:,0][::-1], AB_ij[:,1][::-1], rho, cp)
ABtemp = AB_temp_data.sortby("ocean_time")
ABvel = AB_vel_data.sortby("ocean_time")

# Eddy_AB = ABheat_flux.Eddy_variability(ABtemp,ABvel, track_coords)
Monthly_AB = ABheat_flux.Monthly_variability(ABtemp,ABvel, track_coords)
# Annual_AB = ABheat_flux.Inter_annual_variability(ABtemp,ABvel, track_coords)





#BC heat flux
BC_lon_lat, BC_ij = track_coords.BC_track();

BCheat_flux = HeatFlux(ds, BC_lon_lat, BC_ij[:,0], BC_ij[:,1], rho, cp)
BCtemp = BC_temp_data.sortby("ocean_time")
BCvel = BC_vel_data.sortby("ocean_time")

# Eddy_BC = BCheat_flux.Eddy_variability(BCtemp,BCvel, track_coords)
Monthly_BC = BCheat_flux.Monthly_variability(BCtemp,BCvel, track_coords)
# Annual_BC = BCheat_flux.Inter_annual_variability(BCtemp,BCvel, track_coords)

#CD heat flux
CD_lon_lat, CD_ij = track_coords.CD_track( 0.32,50);

CDheat_flux = HeatFlux(ds, CD_lon_lat, CD_ij[:,0], CD_ij[:,1], rho, cp)
CDtemp = CD_temp_data.sortby("ocean_time")
CDvel = CD_vel_data.sortby("ocean_time")

# Eddy_CD = CDheat_flux.Eddy_variability(CDtemp,CDvel, track_coords)
Monthly_CD = CDheat_flux.Monthly_variability(CDtemp,CDvel, track_coords)
# Annual_CD = CDheat_flux.Inter_annual_variability(CDtemp,CDvel, track_coords)


# Save the values
xr.DataArray(Monthly_AB, name='Monthly_AB').to_netcdf('Monthly_AB.nc')
xr.DataArray(Monthly_BC, name='Monthly_BC').to_netcdf('Monthly_BC.nc')
xr.DataArray(Monthly_CD, name='Monthly_CD').to_netcdf('Monthly_CD.nc')

# xr.DataArray(Annual_AB, name='Annual_AB').to_netcdf('Annual_AB.nc')
# xr.DataArray(Annual_BC, name='Annual_BC').to_netcdf('Annual_BC.nc')
# xr.DataArray(Annual_CD, name='Annual_CD').to_netcdf('Annual_CD.nc')
