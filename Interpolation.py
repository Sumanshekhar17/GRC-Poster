# python script for interpolation class
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import geopy.distance as geo_distance



class interpolation:
    def __init__(self, data, track_isobath, fractional_xi_rho, fractional_eta_rho, track_instance):
        
        self.data = data
        self.track_isobath = track_isobath
        self.ext_track_isobath = None
        self.fractional_xi_rho = fractional_xi_rho
        self.ext_fractional_xi_rho = None
        self.fractional_eta_rho = fractional_eta_rho
        self.ext_fractional_eta_rho = None
        self.track_instance = track_instance

  
    
    def interpolate_data(self, data, fractional_xi_rho, fractional_eta_rho, xi_dim, eta_dim, method='linear'):
        interp_kwargs = {xi_dim: fractional_xi_rho, eta_dim: fractional_eta_rho}
        return data.interp(**interp_kwargs, method=method)
    

    def delta_ijtrk(self, fractional_coordinate):
        fractional_coordinate = np.asarray(fractional_coordinate)
        dItrk = np.diff(fractional_coordinate)
        dItrk = np.append(dItrk, dItrk[-1])
        return dItrk

    def delta_s(self, coordinates):
        s = [geo_distance.distance(coordinates[i], coordinates[i+1]).m for i in range(len(coordinates) - 1)]
        s.append(geo_distance.distance(coordinates[-1], coordinates[-2]).m)
        return np.array(s)

    def compute_vectors(self,delta_I, delta_J, delta_S, pm_trk, pn_trk):
        i_component = (delta_I/delta_S)/pm_trk
        j_component = (delta_J/delta_S)/pn_trk
        p_vector = np.vectorize(complex)(i_component, j_component)
        n_vector = -np.vectorize(complex)(-j_component, i_component)
        return p_vector, n_vector

    def compute_complex_velocity(self, u_trk_slice, v_trk_slice):
        return xr.apply_ufunc(lambda x, y: x + 1j*y, u_trk_slice, v_trk_slice, dask="allowed")

    def compute_velocity(self, complex_velocity, vector):
        complex_velocity_bc, vector_bc = xr.broadcast(complex_velocity, vector)
        return xr.ufuncs.real(complex_velocity_bc * xr.ufuncs.conj(vector_bc))
    
    def dz(self):
        #depth
        dz = self.data.dz  

        dz = dz.where(dz < 1e+35, np.nan)
        dz_trk = self.interpolate_data(dz, self.fractional_xi_rho, self.fractional_eta_rho, 'xi_rho', 'eta_rho')
        
        z_rho = (self.data.z_rho)
        z_rho = z_rho.where(z_rho < 1e+20, np.nan)
        z_rho = self.interpolate_data(z_rho, self.fractional_xi_rho, self.fractional_eta_rho, 'xi_rho', 'eta_rho')
        
        dz_trk['z_rho']   =  z_rho.fillna(0)
        
        return dz_trk
    
    
    
    def extended_coords(self):
        """
        Extend the coordinates for better interpolation at the corner/end values.

        Modifies:
            self.ext_track_isobath
            self.ext_fractional_xi_rho
            self.ext_fractional_eta_rho
        """

        # Initialization
        self.ext_track_isobath = self.track_isobath.copy()
        self.ext_fractional_xi_rho = self.fractional_xi_rho.copy()
        self.ext_fractional_eta_rho = self.fractional_eta_rho.copy()

        # Define end points and slopes
        xi2, yi2 = self.track_isobath[-1]
        xl2, yl2 = self.fractional_xi_rho[-1], self.fractional_eta_rho[-1]

        mi = (yi2 - self.track_isobath[0][1]) / (xi2 - self.track_isobath[0][0])
        ml = (yl2 - self.fractional_eta_rho[0]) / (xl2 - self.fractional_xi_rho[0])

        # Calculate distances
#         ri = geo_distance.distance(self.track_isobath[-1], self.track_isobath[0]).m / 50
        ri = np.sqrt((yi2 - self.track_isobath[0][1])**2 + (xi2 - self.track_isobath[0][0])**2)
        rl = np.sqrt((yl2 - self.fractional_eta_rho[0])**2 + (xl2 - self.fractional_xi_rho[0])**2)

        # Precalculate constant values
        n_values = np.arange(5) + 1
        angle_i = np.arctan(mi)
        angle_l = np.arctan(ml)

        # Calculate extended coordinates

        x_ext_i = float(xi2) + n_values * float(ri) * np.cos(float(angle_i))
        y_ext_i = float(yi2) + n_values * float(ri) * np.sin(float(angle_i))

        x_ext_l = float(xl2) + n_values * float(rl) * np.cos(float(angle_l))
        y_ext_l = float(yl2) + n_values * float(rl) * np.sin(float(angle_l))

        # Update lists
        self.ext_track_isobath = xr.concat([self.ext_track_isobath, xr.DataArray(np.column_stack((x_ext_i, y_ext_i)), dims=["along_track","lon_lat"])], dim="along_track")
        
        self.ext_fractional_xi_rho = xr.concat([self.ext_fractional_xi_rho, xr.DataArray(x_ext_l, dims=["along_track"])], dim="along_track")
        
        self.ext_fractional_eta_rho = xr.concat([self.ext_fractional_eta_rho, xr.DataArray(y_ext_l, dims="along_track")], dim="along_track")
        
        

        
    
    def temperature(self):
        
        """
        Interpolate temperature values along the desired track coordinates.

        This method performs the following steps:
        1. Compute the extended coordinates required for interpolation.
        2. Extract ROMS temperature (`temp`) data from the dataset and mask invalid values.
        3. Interpolate temperature values along the extended fractional rho coordinates.
        4. Extract depth (`z_rho`) values and mask invalid values.
        5. Interpolate depth values along the extended fractional rho coordinates.
        6. Assign the interpolated depth values to the interpolated temperature DataArray.

        Parameters:
        None

        Returns:
        None, but updates the internal state of the object by setting the interpolated temperature and depth values.

        Note:
        The interpolated temperature DataArray has the depth values assigned as an additional coordinate.
        """
        
        self.extended_coords()
        
        
        temp = (self.data).temp

        temp = temp.where(temp < 1e+35, np.nan)
        print(type(temp))

        temp_trk = self.interpolate_data(temp, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')


        z_rhoab = (self.data.z_rho)
        z_rhoab = z_rhoab.where(z_rhoab < 1e+20, np.nan)
        z_rhoab = self.interpolate_data(z_rhoab, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')

        temp_trk['z_rho'] =  z_rhoab.fillna(0)
        
        print("temp_trk time dimension: ", temp_trk.ocean_time.values)
        
        return temp_trk.isel(along_track = slice(None, -5))
    
    def velocity(self):
        
        #vector formation
        """
        Compute the velocity components parallel and normal to the track direction.

        This method involves multiple steps:
        1. Extend the coordinates for interpolation.
        2. Calculate the grid metrics (pm and pn) along the track.
        3. Compute the delta values for I, J, and S along the track.
        4. Determine the vectors (p and n) along the track.
        5. Interpolate the u and v velocity components onto the track.
        6. Compute the complex velocity representation.
        7. Decompose the complex velocity into components parallel (along-track) and normal (cross-track) to the track direction.
        8. Assign depth values to the computed velocity components.

        Parameters:
        None

        Returns:
        tuple:
            V_normal (xarray.DataArray): Normal (cross-track) velocity (m/s) component with associated depth values. 
            V_parallel (xarray.DataArray): Parallel (along-track) velocity (m/s) component with associated depth values. 

        Note:
        The returned DataArrays have depth values assigned as additional coordinates.
        """    
        def rho_velocity(component):
            if component == 'u':
                u_strip = self.data.u.fillna(0)


                u_rho = u_strip.rolling(xi_u=2, center=True).mean()
                u_rho = u_rho.rename({'xi_u':'xi_rho'})
                u_rho = u_rho.isel(eta_rho=slice(1,105), xi_rho=slice(1,241))
                u_rho.coords['z_rho_u'] = self.data.z_rho
                
                rho_velocity = u_rho
                


            elif component == 'v':

                v_strip = self.data.v.fillna(0)

                v_rho = v_strip.rolling(eta_v=2, center=True).mean()
                v_rho = v_rho.rename({'eta_v':'eta_rho'})
                v_rho = v_rho.isel(eta_rho=slice(1,105), xi_rho=slice(1,241))
                v_rho.coords['z_rho_v'] = self.data.z_rho
                
                rho_velocity = v_rho

            else :
                print("ERROR: function rho_velocity() only expects 'u' and 'v' as an argument")

            mask = self.data.mask_rho
            return rho_velocity*(mask.where(mask != 0))
            
            
        self.extended_coords()

        pm = self.data.pm
        pn = self.data.pn
        pm_trk = self.interpolate_data(pm, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')
        pn_trk = self.interpolate_data(pn, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')

        delta_I, delta_J = self.delta_ijtrk(self.ext_fractional_xi_rho), self.delta_ijtrk(self.ext_fractional_eta_rho)
        
#         print("extended_isobath is: ", self.ext_track_isobath)

        delta_S = xr.DataArray(self.delta_s(self.ext_track_isobath), dims=['along_track'])

        p_vector, n_vector = self.compute_vectors(delta_I, delta_J, delta_S, pm_trk, pn_trk)


        #Velocity

        u = rho_velocity('u')
#         u = self.data.u
#         u = u.where(u < 1e+35, np.nan)

        v = rho_velocity('v')

#         v = self.data.v
#         v = v.where(v < 1e+35, np.nan)


#         u_slice = self.interpolate_data(u, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_u', 'eta_rho')
#         v_slice = self.interpolate_data(v, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_v')

        u_slice = self.interpolate_data(u, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')
        v_slice = self.interpolate_data(v, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')
        
        
        u_trk_slice = u_slice
        v_trk_slice = v_slice

        complex_velocity = self.compute_complex_velocity(u_trk_slice, v_trk_slice)
        p_vector = xr.DataArray(p_vector, dims = ["along_track"])
        n_vector = xr.DataArray(n_vector, dims = ["along_track"])

        V_parallel = self.compute_velocity(complex_velocity, p_vector)
        V_normal = self.compute_velocity(complex_velocity, n_vector)

        z_rhou = self.data.z_rho_u
        z_rhou = z_rhou.where(z_rhou < 1e+20, np.nan)
        z_rhou = self.interpolate_data(z_rhou, self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_u', 'eta_rho')


        z_rhov = self.data.z_rho_v
        z_rhov = z_rhov.where(z_rhov < 1e+20, np.nan)
        z_rhov = self.interpolate_data(z_rhov,
                                      self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_v')
        
        z_rho = self.data.z_rho
        z_rho = self.interpolate_data(z_rho,
                                      self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')
        
        lat_rho = self.data.lat_rho
        lat_rho = self.interpolate_data(lat_rho,
                                      self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')

        lon_rho = self.data.lon_rho
        lon_rho = self.interpolate_data(lon_rho,
                                      self.ext_fractional_xi_rho, self.ext_fractional_eta_rho, 'xi_rho', 'eta_rho')

#         V_normal['z_rho_u'], V_normal['z_rho_v'] = z_rhou.fillna(0), z_rhov.fillna(0)
#         V_parallel['z_rho_u'], V_parallel['z_rho_v'] = z_rhou.fillna(0), z_rhov.fillna(0)
        
        V_normal['z_rho'], V_normal['lat_rho'], V_normal['lon_rho'] = z_rho.fillna(0), lat_rho, lon_rho
        V_parallel['z_rho'], V_parallel['lat_rho'], V_parallel['lon_rho'] = z_rho.fillna(0), lat_rho, lon_rho
        
        V_parallel.attrs['long_name'] = 'Parallel component'
        V_normal.attrs['long_name'] = 'Normal Component'
        
        return V_normal.isel(along_track=slice(None, -5)), V_parallel[:-5].isel(along_track=slice(None, -5))




