#python class for heat flux calculation

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from Interpolation import interpolation
from track import track 

class HeatFlux:
    """
    This class is used to calculate the decomposition of heat flux through 
    the face created by the 65 m isobath.

    Attributes:
        - rho (float): "Density of the fluid."
        - cp (float): "Specific heat capacity."
        - data (xarray.DataArray): "ROMS data."
        - track_isobath (xarray.DataArray): "coordinates of 65m isobath track in lon, lat."
        - fractional_xi_rho (xarray.DataArray): "i component of coordinates of 65m isobath track in fractional grid indices."
        - fractional_eta_rho: "j component of coordinates of 65m isobath track in fractional grid indices."

    Example:
        >>> instance = HeatFlux(rho_value, cp_value)
        Heat Flux calculation instance is created.
    """

    
    def __init__(self, data, track_isobath, fractional_xi_rho, fractional_eta_rho, rho, cp):
        
        #initialization variables
        
        self.data = data # don't put comma at the end of the assaignment within __init__ method, you will end up creating tuple 
        self.track_isobath = track_isobath
        self.fractional_xi_rho = fractional_xi_rho
        self.fractional_eta_rho = fractional_eta_rho
        self.rho = rho
        self.cp = cp
        self.dz = None
        self.ds = None
        
        
        print("Heat Flux calculation instance is created.")
        
    
    def total(self, temp, vel, track_instance):
        """
        This method calculates the total heat flux using hourly temperature and hourly velocity along the track.

        Parameters:
            temp (xarray.DataArray): 
                Interpolated hourly temperature along the track.
            vel (xarray.DataArray): 
                Interpolated hourly normal velocity along the track.
            dz (xarray.DataArray): 
                Length of vertical cells at each point along the track.
            ds (xarray.DataArray): 
                Length of horizontal cells between each point along the track.

        Returns:
            xarray.DataArray: Total heat flux across the face.
        """

        interp = interpolation(self.data, self.track_isobath, self.fractional_xi_rho, self.fractional_eta_rho, track_instance)
        self.dz = (interp.dz()).mean(dim="ocean_time")
        self.ds = xr.DataArray(interp.delta_s(self.track_isobath), dims=['along_track'])
        
        Total_heat_flux = ((self.rho*self.cp*((temp*vel).mean(dim="ocean_time")))*self.dz*self.ds).sum()
        
        return Total_heat_flux
    
    def Annual_variability(self, temp, vel, track_instance):
        """
        This method calculates the heat flux due to annual variability using hourly temperature and hourly velocity along the track.

        Parameters:
            temp (xarray.DataArray): 
                Interpolated hourly temperature along the track.
            vel (xarray.DataArray): 
                Interpolated hourly normal velocity along the track.
            dz (xarray.DataArray): 
                Length of vertical cells at each point along the track.
            ds (xarray.DataArray): 
                Length of horizontal cells between each point along the track.

        Returns:
            xarray.DataArray: Heat flux due to annual variability across the face.
        """
        interp = interpolation(self.data, self.track_isobath, self.fractional_xi_rho, self.fractional_eta_rho, track_instance)
        
        self.dz = (interp.dz()).mean(dim="ocean_time")
        self.ds =  xr.DataArray(interp.delta_s(self.track_isobath), dims=['along_track'])
        
        Va = (vel.resample(ocean_time='1Y')).mean(dim="ocean_time")
        Ta = (temp.resample(ocean_time='1Y')).mean(dim="ocean_time")   
        
        Heat_annual = ((self.rho*self.cp*((Ta*Va).mean(dim="ocean_time")))*self.dz*self.ds).sum()
        
        return Heat_annual
    
    
    
    def Inter_annual_variability(self, temp, vel, track_instance):
        
        interp = interpolation(self.data, self.track_isobath, self.fractional_xi_rho, self.fractional_eta_rho, track_instance)
        
        self.dz = (interp.dz()).mean(dim="ocean_time")
        self.ds =  xr.DataArray(interp.delta_s(self.track_isobath), dims=['along_track'])
        
        Vt = vel.mean(dim="ocean_time")
        Tt = temp.mean(dim="ocean_time")
        
        
        Va = (vel.resample(ocean_time='1Y')).mean(dim="ocean_time") - Vt.values
        Ta = (temp.resample(ocean_time='1Y')).mean(dim="ocean_time") - Tt.values  
        
        Heat_inter_annual = ((self.rho*self.cp*((Ta*Va).mean(dim="ocean_time")))*self.dz*self.ds).sum()
        
        return Heat_inter_annual
    
    
    def Monthly_variability(self, temp, vel, track_instance):
        
        """
        This method calculates the heat flux due to monthly variability using hourly temperature and hourly velocity along the track.

        Parameters:
            temp (xarray.DataArray): 
                Interpolated hourly temperature along the track.
            vel (xarray.DataArray): 
                Interpolated hourly normal velocity along the track.
            dz (xarray.DataArray): 
                Length of vertical cells at each point along the track.
            ds (xarray.DataArray): 
                Length of horizontal cells between each point along the track.

        Returns:
            xarray.DataArray: Heat flux due to monthly variability across the face.
        """
        
        interp = interpolation(self.data, self.track_isobath, self.fractional_xi_rho, self.fractional_eta_rho, track_instance)

        self.dz = (interp.dz()).mean(dim="ocean_time")
        self.ds =  xr.DataArray(interp.delta_s(self.track_isobath), dims=['along_track'])
        
        Va = (vel.resample(ocean_time='1Y')).mean(dim="ocean_time")
        Ta = (temp.resample(ocean_time='1Y')).mean(dim="ocean_time")
        
        
        Vm = (vel.resample(ocean_time='1M')).mean(dim="ocean_time")
        Tm = (temp.resample(ocean_time='1M')).mean(dim="ocean_time")        
        
        
#         T_variability = Tm.groupby('ocean_time.year').map(lambda x: x - Ta.sel(ocean_time=x['ocean_time.year'][0]))
#         V_variability = Vm.groupby('ocean_time.year').map(lambda x: x - Va.sel(ocean_time=x['ocean_time.year'][0]))

        def subtract_annual_mean(monthly_data, annual_data):
        
            year = monthly_data['ocean_time.year'].values[0]
            return monthly_data - annual_data.sel(year=year)

        T_variability = Tm.groupby('ocean_time.year').map(subtract_annual_mean, args=(Ta,))
        V_variability = Vm.groupby('ocean_time.year').map(subtract_annual_mean, args=(Va,))


        
        Heat_monthly = ((self.rho*self.cp*((T_variability*V_variability).mean(dim="ocean_time")))*self.dz*self.ds).sum()
        
        return Heat_monthly
    
    def Eddy_variability(self, temp, vel, track_instance):
        
        
        """
        This method calculates the heat flux due to eddy variability using hourly temperature and hourly velocity along the track.

        Parameters:
            temp (xarray.DataArray): 
                Interpolated hourly temperature along the track.
            vel (xarray.DataArray): 
                Interpolated hourly normal velocity along the track.
            dz (xarray.DataArray): 
                Length of vertical cells at each point along the track.
            ds (xarray.DataArray): 
                Length of horizontal cells between each point along the track.

        Returns:
            xarray.DataArray: Heat flux due to eddy variability across the face.
        """

        interp = interpolation(self.data, self.track_isobath, self.fractional_xi_rho, self.fractional_eta_rho, track_instance)
        


        self.dz = (interp.dz()).mean(dim="ocean_time")
        self.ds =  xr.DataArray(interp.delta_s(self.track_isobath), dims=['along_track'])
        
        vel = vel.sortby('ocean_time')
        temp = temp.sortby('ocean_time')
        Vm = (vel.resample(ocean_time='1M')).mean(dim="ocean_time")
        Tm = (temp.resample(ocean_time='1M')).mean(dim="ocean_time")
        
        V_variability = vel.groupby('ocean_time.month').map(lambda x: x - Vm.isel(ocean_time=x['ocean_time.month'][0]-1))
        T_variability = temp.groupby('ocean_time.month').map(lambda x: x - Tm.isel(ocean_time=x['ocean_time.month'][0]-1)) 
        
        Heat_eddy = ((self.rho*self.cp*((T_variability*V_variability).mean(dim="ocean_time")))*self.dz*self.ds).sum()
        
        return Heat_eddy
        
        
        
        
        
        
        