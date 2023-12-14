import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import xroms
import dask
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

link = "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/DopAnV3R3-ini2007_da/mon_avg"
chunks = {'ocean_time' : 20}
ds = xr.open_dataset(link, chunks = chunks)

ds, xgrid = xroms.roms_dataset(ds)
isobath = xr.open_dataset('./isobath_coordinate2.nc')


# start and end time
start = '2020-01-01'
end = '2020-12-31'

ds = ds.sel(ocean_time=slice(start,end))

##functions
def interpolate_data(data, fractional_xi_rho, fractional_eta_rho, xi_dim, eta_dim, method='linear'):
    interp_kwargs = {xi_dim: fractional_xi_rho, eta_dim: fractional_eta_rho}
    return data.interp(**interp_kwargs, method=method)

# @dask.delayed
def delta_ijtrk(fractional_coordinate):
    fractional_coordinate = np.asarray(fractional_coordinate)
    dItrk = np.diff(fractional_coordinate)
    dItrk = np.append(dItrk, dItrk[-1])
    return dItrk

import geopy.distance as geo_distance

# @dask.delayed
def delta_s(coordinates):
    
    s = [geo_distance.distance(coordinates[i], coordinates[i+1]).m for i in range(len(coordinates) - 1)]
    s.append(geo_distance.distance(coordinates[-1], coordinates[-2]).m)
    return np.array(s)

# @dask.delayed
def compute_vectors(delta_I, delta_J, delta_S, pm_trk, pn_trk):
    i_component = (delta_I/delta_S)/pm_trk
    j_component = (delta_J/delta_S)/pn_trk
    p_vector = np.vectorize(complex)(i_component, j_component)
    n_vector = -np.vectorize(complex)(-j_component, i_component)
    return p_vector, n_vector

# @dask.delayed
def compute_complex_velocity(u_trk_slice, v_trk_slice):
    return xr.apply_ufunc(lambda x, y: x + 1j*y, u_trk_slice, v_trk_slice, dask="allowed")

# @dask.delayed
def compute_velocity(complex_velocity, vector):
    complex_velocity_bc, vector_bc = xr.broadcast(complex_velocity, vector)
    return xr.ufuncs.real(complex_velocity_bc * xr.ufuncs.conj(vector_bc))


##computations

CD_ij_isobath, CD_isobath = isobath.coords_ij[442:], isobath.coords_lon_lat[442:]

fractional_xi_rho, fractional_eta_rho = CD_ij_isobath[:,0], CD_ij_isobath[:,1]

#temperature
temp = ds.temp
#depth
dz = ds.dz

CD_dz_trk = interpolate_data(dz, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')
CD_temp_trk = interpolate_data(temp, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')

#vector formation

pm = ds.pm
pn = ds.pn
pm_trk = interpolate_data(pm, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')
pn_trk = interpolate_data(pn, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')

delta_I, delta_J = delta_ijtrk(fractional_xi_rho), delta_ijtrk(fractional_eta_rho)

CD_delta_S = xr.DataArray(delta_s(CD_isobath), dims=['along_track'])

p_vector, n_vector = compute_vectors(delta_I, delta_J, CD_delta_S, pm_trk, pn_trk)


#Velocity
u = ds.u
v = ds.v


u_slice = interpolate_data(u, fractional_xi_rho, fractional_eta_rho, 'xi_u', 'eta_rho')
v_slice = interpolate_data(v, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_v')

u_trk_slice = u_slice
v_trk_slice = v_slice

complex_velocity = compute_complex_velocity(u_trk_slice, v_trk_slice)
p_vector = xr.DataArray(p_vector, dims = ["along_track"])
n_vector = xr.DataArray(n_vector, dims = ["along_track"])

CD_V_parallel = compute_velocity(complex_velocity, p_vector)
CD_V_normal = compute_velocity(complex_velocity, n_vector)


CD_V_parallel.attrs['long_name'] = 'Parallel component'
CD_V_normal.attrs['long_name'] = 'Normal Component'


CD_Tm_spatial = ((CD_temp_trk*CD_dz_trk*CD_delta_S).sum(dim=('along_track','s_rho')))/(CD_dz_trk*CD_delta_S).sum(dim=('along_track','s_rho'))


ds_combined = xr.Dataset({
    'CD_face_temp': CD_temp_trk,
    'CD_V_normal': CD_V_normal,
    'CD_V_parallel': CD_V_parallel,
    'CD_Tm_spatial': CD_Tm_spatial
    })

for variable in ds_combined.data_vars:
    ds_combined[variable].attrs.pop('grid', None)

ds_combined.chunk({'ocean_time':100}).to_netcdf('./CD_2020_monthly.nc')


#######BC
BC_ij_isobath, BC_isobath = isobath.coords_ij[45:442], isobath.coords_lon_lat[45:442]
fractional_xi_rho, fractional_eta_rho = BC_ij_isobath[:,0], BC_ij_isobath[:,1]



#temperature
temp = ds.temp
#depth
dz = ds.dz

BC_dz_trk = interpolate_data(dz, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')
BC_temp_trk = interpolate_data(temp, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')




#vector formation

pm = ds.pm
pn = ds.pn
pm_trk = interpolate_data(pm, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')
pn_trk = interpolate_data(pn, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')

delta_I, delta_J = delta_ijtrk(fractional_xi_rho), delta_ijtrk(fractional_eta_rho)



BC_delta_S = xr.DataArray(delta_s(BC_isobath), dims=['along_track'])

p_vector, n_vector = compute_vectors(delta_I, delta_J, BC_delta_S, pm_trk, pn_trk)



#Velocity
u = ds.u
v = ds.v


u_slice = interpolate_data(u, fractional_xi_rho, fractional_eta_rho, 'xi_u', 'eta_rho')
v_slice = interpolate_data(v, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_v')

u_trk_slice = u_slice
v_trk_slice = v_slice

complex_velocity = compute_complex_velocity(u_trk_slice, v_trk_slice)
p_vector = xr.DataArray(p_vector, dims = ["along_track"])
n_vector = xr.DataArray(n_vector, dims = ["along_track"])

BC_V_parallel = compute_velocity(complex_velocity, p_vector)
BC_V_normal = compute_velocity(complex_velocity, n_vector)


BC_V_parallel.attrs['long_name'] = 'Parallel component'
BC_V_normal.attrs['long_name'] = 'Normal Component'


BC_Tm_spatial = ((BC_temp_trk*BC_dz_trk*BC_delta_S).sum(dim=('along_track','s_rho')))/(BC_dz_trk*BC_delta_S).sum(dim=('along_track','s_rho'))



ds_combined = xr.Dataset({
    'BC_face_temp': BC_temp_trk,
    'BC_V_normal': BC_V_normal,
    'BC_V_parallel': BC_V_parallel,
    'BC_Tm_spatial': BC_Tm_spatial
    })

for variable in ds_combined.data_vars:
    ds_combined[variable].attrs.pop('grid', None)

ds_combined.chunk({'ocean_time':100}).to_netcdf('./BC_2020_monthly.nc')



#######AB

AB_ij_isobath, AB_isobath = isobath.coords_ij[0:45], isobath.coords_lon_lat[0:45]

fractional_xi_rho, fractional_eta_rho = AB_ij_isobath[:,0], AB_ij_isobath[:,1]




#temperature
temp = ds.temp
#depth
dz = ds.dz

AB_dz_trk = interpolate_data(dz, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')
AB_temp_trk = interpolate_data(temp, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')




#vector formation

pm = ds.pm
pn = ds.pn
pm_trk = interpolate_data(pm, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')
pn_trk = interpolate_data(pn, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_rho')

delta_I, delta_J = delta_ijtrk(fractional_xi_rho), delta_ijtrk(fractional_eta_rho)



AB_delta_S = xr.DataArray(delta_s(AB_isobath), dims=['along_track'])

p_vector, n_vector = compute_vectors(delta_I, delta_J, AB_delta_S, pm_trk, pn_trk)



#Velocity
u = ds.u
v = ds.v


u_slice = interpolate_data(u, fractional_xi_rho, fractional_eta_rho, 'xi_u', 'eta_rho')
v_slice = interpolate_data(v, fractional_xi_rho, fractional_eta_rho, 'xi_rho', 'eta_v')

u_trk_slice = u_slice
v_trk_slice = v_slice

complex_velocity = compute_complex_velocity(u_trk_slice, v_trk_slice)
p_vector = xr.DataArray(p_vector, dims = ["along_track"])
n_vector = xr.DataArray(n_vector, dims = ["along_track"])

AB_V_parallel = compute_velocity(complex_velocity, p_vector)
AB_V_normal = compute_velocity(complex_velocity, n_vector)


AB_V_parallel.attrs['long_name'] = 'Parallel component'
AB_V_normal.attrs['long_name'] = 'Normal Component'


AB_Tm_spatial = ((AB_temp_trk*AB_dz_trk*AB_delta_S).sum(dim=('along_track','s_rho')))/(AB_dz_trk*AB_delta_S).sum(dim=('along_track','s_rho'))



ds_combined = xr.Dataset({
    'AB_face_temp': AB_temp_trk,
    'AB_V_normal': AB_V_normal,
    'AB_V_parallel': AB_V_parallel,
    'AB_Tm_spatial': AB_Tm_spatial
    })

for variable in ds_combined.data_vars:
    ds_combined[variable].attrs.pop('grid', None)

ds_combined.chunk({'ocean_time':100}).to_netcdf('./AB_2020_monthly.nc')



