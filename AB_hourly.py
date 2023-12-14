import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import xroms
import dask
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

link = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/his"
chunks = {'ocean_time' : 20}
ds = xr.open_dataset(link, chunks = chunks)

ds, xgrid = xroms.roms_dataset(ds)
isobath = xr.open_dataset('./isobath_coordinate2.nc')


# start and end time
start = '2020-01-01'
end = '2020-12-31'


#track coordinates

AB_ij_isobath, AB_isobath = isobath.coords_ij[0:45], isobath.coords_lon_lat[0:45]

fractional_xi_rho, fractional_eta_rho = AB_ij_isobath[:,0], AB_ij_isobath[:,1]


#functions used in notebook

def interpolate_data(data, fractional_xi_rho, fractional_eta_rho, xi_dim, eta_dim, method='linear'):
    interp_kwargs = {xi_dim: fractional_xi_rho, eta_dim: fractional_eta_rho}
    return data.interp(**interp_kwargs, method=method)

def delta_ijtrk(fractional_coordinate):
    fractional_coordinate = np.asarray(fractional_coordinate)
    dItrk = np.diff(fractional_coordinate)
    dItrk = np.append(dItrk, dItrk[-1])
    return dItrk

import geopy.distance as geo_distance

def delta_s(coordinates):
    
    s = [geo_distance.distance(coordinates[i], coordinates[i+1]).m for i in range(len(coordinates) - 1)]
    s.append(geo_distance.distance(coordinates[-1], coordinates[-2]).m)
    return np.array(s)

def compute_vectors(delta_I, delta_J, AB_delta_S, pm_trk, pn_trk):
    i_component = (delta_I/AB_delta_S)/pm_trk
    j_component = (delta_J/AB_delta_S)/pn_trk
    p_vector = np.vectorize(complex)(i_component, j_component)
    n_vector = -np.vectorize(complex)(-j_component, i_component)
    return p_vector, n_vector

def compute_complex_velocity(u_trk_slice, v_trk_slice):
    return xr.apply_ufunc(lambda x, y: x + 1j*y, u_trk_slice, v_trk_slice, dask="allowed")

def compute_velocity(complex_velocity, vector):
    complex_velocity_bc, vector_bc = xr.broadcast(complex_velocity, vector)
    return xr.ufuncs.real(complex_velocity_bc * xr.ufuncs.conj(vector_bc))




#temperature
temp = ds.temp.sel(ocean_time=slice(start,end))

temp = temp.where(temp < 1e+35, np.nan)

#depth
dz = ds.dz.sel(ocean_time=slice(start, end))  

dz = dz.where(dz < 1e+35, np.nan)


AB_dz_trk = interpolate_data(dz, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')
AB_temp_trk = interpolate_data(temp, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')


z_rhoab = (ds.z_rho.sel(ocean_time= slice(start,end)))
z_rhoab = z_rhoab.where(z_rhoab < 1e+20, np.nan)


z_rhoab = interpolate_data(z_rhoab, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')

AB_dz_trk['z_rho']   =  z_rhoab.fillna(0)
AB_temp_trk['z_rho'] =  z_rhoab.fillna(0)


#vector formation

pm = ds.pm
pn = ds.pn
pm_trk = interpolate_data(pm, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')
pn_trk = interpolate_data(pn, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')

delta_I, delta_J = delta_ijtrk(fractional_xi_rho), delta_ijtrk(fractional_eta_rho)

AB_delta_S = xr.DataArray(delta_s(AB_isobath), dims=['along_track'])

p_vector, n_vector = compute_vectors(delta_I, delta_J, AB_delta_S, pm_trk, pn_trk)


#Velocity


u = ds.u.sel(ocean_time=slice(start,end))
u = u.where(u < 1e+35, np.nan)

v = ds.v.sel(ocean_time=slice(start,end))
v = v.where(v < 1e+35, np.nan)


u_slice = interpolate_data(u, fractional_xi_rho, fractional_eta_rho, 'xi_u', 'eta_rho')
v_slice = interpolate_data(v, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_v')

u_trk_slice = u_slice
v_trk_slice = v_slice

complex_velocity = compute_complex_velocity(u_trk_slice, v_trk_slice)
p_vector = xr.DataArray(p_vector, dims = ["along_track"])
n_vector = xr.DataArray(n_vector, dims = ["along_track"])

AB_V_parallel = compute_velocity(complex_velocity, p_vector)
AB_V_normal = compute_velocity(complex_velocity, n_vector)

z_rhou_ab = ds.z_rho_u.sel(ocean_time=slice(start,end))
z_rhou_ab = z_rhou_ab.where(z_rhou_ab < 1e+20, np.nan)
z_rhou_ab = interpolate_data(z_rhou_ab, fractional_xi_rho, fractional_eta_rho, 'xi_u', 'eta_rho')


z_rhov_ab = ds.z_rho_v.sel(ocean_time=slice(start,end))
z_rhov_ab = z_rhov_ab.where(z_rhov_ab < 1e+20, np.nan)
z_rhov_ab = interpolate_data(z_rhov_ab
                         , fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_v')

z_rhou_cd = (ds.z_rho_u.sel(ocean_time=slice(start,end))*ds.mask_u)
z_rhou_cd = interpolate_data(z_rhou_cd, fractional_xi_rho, fractional_eta_rho, 'xi_u', 'eta_rho')


AB_V_normal['z_rho_u'], AB_V_normal['z_rho_v'] = z_rhou_ab.fillna(0), z_rhov_ab.fillna(0)
AB_V_parallel['z_rho_u'], AB_V_parallel['z_rho_v'] = z_rhou_ab.fillna(0), z_rhov_ab.fillna(0)
AB_V_parallel.attrs['long_name'] = 'Parallel component'
AB_V_normal.attrs['long_name'] = 'Normal Component'


ds_combined = xr.Dataset({
    'AB_face_temp': AB_temp_trk,
    'AB_V_normal': AB_V_normal,
    'AB_V_parallel': AB_V_parallel,
    'AB_dz_track': AB_dz_trk,
    'AB_ds_track': AB_delta_S
    })

for variable in ds_combined.data_vars:
    ds_combined[variable].attrs.pop('grid', None)

ds_combined.chunk({'ocean_time':100}).to_netcdf('./AB_2020_hourly.nc')






