#importing neccessary files and packages
import xarray as xr
import numpy as np
import xroms
import matplotlib.pyplot as plt
import sys
sys.path.append('/scratch/ss4338/2020/')
from track import track

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#importing hourly doppio data
link = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/his"
chunk = {"ocean_time":10}
ds = xroms.open_netcdf(link, chunks=chunk)

#land mask
land_mask = ds.mask_rho

#generating 65m isobath track
bathymetry = ds.h
extent = [-76,-65, 34, 44]
sigma = 15
lon_left = -75
lon_right = -70.65
isobath_value = 65
track_coords = track(ds, bathymetry, extent, lon_left, lon_right, isobath_value, sigma)

AB_lon_lat, AB_ij = track_coords.AB_track(-0.74,50);
CD_lon_lat, CD_ij = track_coords.CD_track( 0.32,50);
BC_lon_lat, BC_ij = track_coords.BC_track();

final_ij_isobath = np.vstack((AB_ij[::-1], BC_ij, CD_ij))


#creating mask according to the geometry of our domain
x1,y1 = AB_ij[0]
x2,y2 = AB_ij[-1]

m1 = (y2-y1)/(x2-x1)
y = y2-m1*x2

x3,y3 = CD_ij[0]
x4,y4 = CD_ij[-1]

m2 = (y4-y3)/(x4-x3)
x = (106 - y4)/m2 + x4


vertices1 = (0, y)  
vertices2 = (x,106)


import matplotlib.path as mpath

def create_mask(shape, polygon):
    y,x = np.mgrid[:shape[0], :shape[1]]
    points = np.column_stack((x.ravel(), y.ravel()))
    path = mpath.Path(polygon)
    mask = path.contains_points(points).reshape(shape)
    
    return mask

shape = (106, 242)
polygon = np.vstack((vertices1, final_ij_isobath, vertices2))



mask = create_mask(shape, polygon)


new_mask = (ds.mask_rho*mask)


#calculation of net sea-flux

netsflux = new_mask*ds.shflux.isel(ocean_time=1000)
area = new_mask*ds.dA
surface_heat = (netsflux*area).sum(dim=['eta_rho','xi_rho'])

surface_heat.attrs.pop('grid', None)
(xr.DataArray(final_ij_isobath)).to_dataset(name='isobath').to_netcdf('./ij_isobath.nc')
surface_heat.to_dataset(name='surface_heat').to_netcdf('/scratch/jwilkin/air-sea-flux.nc')
