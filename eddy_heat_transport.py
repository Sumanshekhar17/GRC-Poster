import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import xroms

linkab = './AB_2020_hourly.nc'
linkbc = './BC_2020_hourly.nc'
linkcd = './CD_2020_hourly.nc'
chunks = {'ocean_time': 20}

AB = xr.open_dataset(linkab, chunks = chunks)
BC = xr.open_dataset(linkbc, chunks = chunks)
CD = xr.open_dataset(linkcd, chunks = chunks)

#AB

AB_monthly = xr.open_dataset('./AB_2020_monthly.nc', chunks = 20)
Vhab = AB.AB_V_normal
Vmab = AB_monthly.AB_V_normal

Thab = AB.AB_face_temp
Tmab = AB_monthly.AB_temp_trk

Vh_monthly_mean = Vmab
Vh_monthly_broadcasted = Vhab.groupby('ocean_time.month').map(lambda x: x - (Vmab.isel(ocean_time=slice(0,13))).isel(ocean_time=x['ocean_time.month'].values[0]))
eddy = Vh_monthly_broadcasted

Th_monthly_mean = Tmab
Th_monthly_broadcasted = Thab.groupby('ocean_time.month').map(lambda x: x - (Tmab.isel(ocean_time=slice(0,13))).isel(ocean_time=x['ocean_time.month'].values[0]))
theta = Th_monthly_broadcasted

eddy.coords['along_track'] = Vhab.along_track
eddy.coords['z_rho_u'] = Vhab.z_rho_u
eddy.coords['z_rho_v'] = Vhab.z_rho_v

theta.coords['along_track'] = Thab.along_track
theta.coords['z_rho'] = Thab.z_rho

AB_eddy_variability = theta*eddy

#CD

CD_monthly = xr.open_dataset('./CD_2020_monthly.nc', chunks = 20)

Vhcd = CD.CD_V_normal
Vmcd = CD_monthly.CD_V_normal

Vh_monthly_mean = Vmcd
Vh_monthly_broadcasted = Vhcd.groupby('ocean_time.month').map(lambda x: x - (Vmcd.isel(ocean_time=slice(0,13))).isel(ocean_time=x['ocean_time.month'].values[0]))
eddycd = Vh_monthly_broadcasted

eddycd.coords['along_track'] = Vhcd.along_track
eddycd.coords['z_rho_u'] = Vhcd.z_rho_u
eddycd.coords['z_rho_v'] = Vhcd.z_rho_v

Thcd = CD.CD_face_temp
Tmcd = CD_monthly.CD_temp_trk

Th_monthly_mean = Tmcd
Th_monthly_broadcasted = Thcd.groupby('ocean_time.month').map(lambda x: x - (Tmcd.isel(ocean_time=slice(0,13))).isel(ocean_time=x['ocean_time.month'].values[0]))
thetacd = Th_monthly_broadcasted

thetacd.coords['along_track'] = Thcd.along_track
thetacd.coords['z_rho'] = Thcd.z_rho

CD_eddy_variability = thetacd*eddycd

#BC

BC_monthly = xr.open_dataset('./BC_2020_monthly.nc', chunks = 20)

Vhbc = BC.BC_V_normal
Vmbc = BC_monthly.BC_V_normal

Vh_monthly_mean = Vmbc
Vh_monthly_broadcasted = Vhbc.groupby('ocean_time.month').map(lambda x: x - (Vmbc.isel(ocean_time=slice(0,13))).isel(ocean_time=x['ocean_time.month'].values[0]))
eddybc = Vh_monthly_broadcasted

eddybc.coords['along_track'] = Vhbc.along_track
eddybc.coords['z_rho_u'] = Vhbc.z_rho_u
eddybc.coords['z_rho_v'] = Vhbc.z_rho_v

Thbc = BC.BC_face_temp
Tmbc = BC_monthly.BC_temp_trk

Th_monthly_mean = Tmbc
Th_monthly_broadcasted = Thbc.groupby('ocean_time.month').map(lambda x: x - (Tmbc.isel(ocean_time=slice(0,13))).isel(ocean_time=x['ocean_time.month'].values[0]))
thetabc = Th_monthly_broadcasted

thetabc.coords['along_track'] = Thbc.along_track
thetabc.coords['z_rho'] = Thbc.z_rho

BC_eddy_variability = thetabc*eddybc


#Heat transport calculation
#AB
Tmsab = AB.AB_Tm_spatial

Ts_monthly_mean = Tmsab
Th_monthly_broadcasted = theta.groupby('ocean_time.month').map(lambda x: x - (Tmsab.isel(ocean_time=x['ocean_time.month'].values[0])))

T_dash = Th_monthly_broadcasted

rho = 1027.8
cp = 3850
AB_eddy_flux = cp*rho*(eddy*T_dash*(AB.AB_dz_track)*(AB.AB_ds_track)).sum(dim=('along_track','s_rho'))

#BC

Tmsbc = BC.BC_Tm_spatial
Ts_monthly_meanbc = Tmsbc
Th_monthly_broadcastedbc = thetabc.groupby('ocean_time.month').map(lambda x: x - (Tmsbc.isel(ocean_time=x['ocean_time.month'].values[0])))
T_dashbc = Th_monthly_broadcastedbc

BC_eddy_flux = cp*rho*(eddybc*T_dashbc*(BC.BC_dz_track)*(BC.BC_ds_track)).sum(dim=('along_track','s_rho'))

#CD

Tmscd = CD.CD_Tm_spatial
Ts_monthly_meancd = Tmscd
Th_monthly_broadcastedcd = thetacd.groupby('ocean_time.month').map(lambda x: x - (Tmscd.isel(ocean_time=x['ocean_time.month'].values[0])))
T_dashcd = Th_monthly_broadcastedcd

CD_eddy_flux = cp*rho*(eddycd.isel(along_track=slice(0,-1))*T_dashcd.isel(along_track=slice(0,-1))*(CD.CD_dz_track)*(CD.CD_ds_track)).sum(dim=('along_track','s_rho'))


eds = xr.Dataset({
    'AB_eddy_heat_transport':AB_eddy_flux,
    'BC_eddy_heat_transport':BC_eddy_flux,
    'CD_eddy_heat_transport':CD_eddy_flux
    
})

for variable in eds.data_vars:
    eds[variable].attrs.pop('grid', None)

eds.to_netcdf('./Eddy_transport_2020_hourly.nc')

