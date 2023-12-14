import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata



#defining a class
class track:
    def __init__(self, ds, bathymetry, extent, lon_left, lon_right, isobath_value, sigma):
        
        """
        Class for representing a track based on a specific isobath.
        
        The track class utilizes batymetry data and a specified isobath value
        to determine the desired track. This track act as the outer off-shore
        boundary of the control volume. It also provides utilities for smoothing
        the track based on a Gaussian filter with a specified sigma.
        
        You can also filter the track by passing the longitude extents.
        
        Parameters:
        ---------------------
            ds (Xarray.Dataset): Dataset containing the necessary variables and coordinates
                                 for track calculation.

            bathymetry (Xarray.Dataset): 2D DataArray containing bathymetry data with dimensions 
                                         corresponding to latitude and longitude.

            extent (tuple ot list): A list or tuple specifying the spatial extent of the data 
                                    in the form [lon_min, lon_max, lat_min, lat_max]

            lon_left (float): The leftmost longitude boundary for the isobath track calculation.

            lon_right (float): The rightmost longitude boundary for the isobath track calculation.

            isobath_value (float): Depth value used to identify the isobath.

            sigma (float): Standard deviation for the Gaussian filter applied to smooth 
                           the isobath track.
                       
                       
        Example:
        -------
        >>> track_instance = track(ds, bathymetry, [-72, -70, 40, 42], -71.5, -70.5, 65, 1.0)
        """

        
        self.ds = ds
        self.bathymetry = bathymetry
        self.extent = extent
        self.sigma = sigma
        self.lon_left = lon_left
        self.lon_right = lon_right
        self.isobath_value = isobath_value
        self.AB_lon_lat = None
        self.AB_ij = None
        self.BC_lon_lat = None
        self.BC_ij = None
        self.CD_lon_lat = None
        self.CD_ij = None
        self.smoothed_isobath_coords = None

        
        
    
    def display(self):
        print(f"We have bathymetry which is {self.bathymetry.attrs} and with shape {self.bathymetry.shape}")
        print(f"Track extent is {self.extent}")
        

    def BC_track(self):
        
        """
        This method is to generate track with required isobath within required longitude constraints.
        
        Parameter: None
        
        Results:
            BC_lon_lat (xarray.DataArray): coordinates in lon and lat format.
            BC_ij (xarray.DataArray): coordinates in fractional xi_rhi and eta_rho format.
        

        """
        
        bathymetry_track = self.bathymetry
        projection = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize = (10,6), subplot_kw = {"projection" : projection})
        
        # set the extent of the map (based on your area of interest)
        ax.set_extent(self.extent)
        
        isobath = bathymetry_track.plot.contour(ax=ax, x='lon_rho', y='lat_rho', levels=[self.isobath_value], transform= projection)
        plt.show()

        path = isobath.collections[0].get_paths()
        path_coords = path[0].vertices
        path_coords = np.flip(path_coords)
        path_coords[:,[0,1]] = path_coords[:,[1,0]]
        
        
        #searching in numpy array
        indices = np.where((self.lon_left <= path_coords[:,0]) & (path_coords[:,0] <= self.lon_right))
        path_coords = path_coords[indices]
        
        
        
        # Convert i and j indices to complex numbers
        complex_coords = path_coords[:, 1] + 1j * path_coords[:, 0]

        # Apply Gaussian filter on the complex coordinates
        sigma = self.sigma
        smoothed_real_coords = gaussian_filter(complex_coords.real, sigma)
        smoothed_imag_coords = gaussian_filter(complex_coords.imag, sigma)


        # Separate the real and imaginary parts back into i and j indices
        smoothed_isobath_coords = np.zeros_like(path_coords)
        smoothed_isobath_coords[:, 1] = smoothed_real_coords #first column is xi_rho
        smoothed_isobath_coords[:, 0] = smoothed_imag_coords #second column is eta_rho
        
        BC_lon_lat = np.column_stack((smoothed_isobath_coords[:, 0], smoothed_isobath_coords[:, 1]))
        BC_lon_lat = xr.DataArray(BC_lon_lat)
        BC_lon_lat = BC_lon_lat.rename({"dim_0":"along_track","dim_1":"lon_lat"})
        
        
        # xi_rho and eta_rho coordinates
        # Replace lat_rho and lon_rho with your actual DataArrays
        lat_rho = self.ds.lat_rho
        lon_rho = self.ds.lon_rho 

        # Replace these with your actual transect coordinates
        transect_lat_rho = smoothed_isobath_coords[:, 1]
        transect_lon_rho = smoothed_isobath_coords[:, 0]

        # Create a 2D meshgrid for xi_rho and eta_rho
        xi_rho  = np.arange(lat_rho.shape[1])
        eta_rho = np.arange(lat_rho.shape[0])
        xi_rho_2d, eta_rho_2d = np.meshgrid(xi_rho, eta_rho)


        # Flatten the 2D arrays
        lat_rho_flat = lat_rho.data.flatten()
        lon_rho_flat = lon_rho.data.flatten()
        xi_rho_flat = xi_rho_2d.flatten()
        eta_rho_flat = eta_rho_2d.flatten()

        # Combine the flattened arrays into a single array
        input_coords = np.column_stack((lat_rho_flat, lon_rho_flat))

        # Interpolate the xi_rho and eta_rho values for the given transect coordinates
        transect_coords = np.column_stack((transect_lat_rho, transect_lon_rho))
        fractional_xi_rho  = griddata(input_coords, xi_rho_flat, transect_coords, method="linear")
        fractional_eta_rho = griddata(input_coords, eta_rho_flat, transect_coords, method="linear")
        
        BC_ij = np.column_stack((fractional_xi_rho, fractional_eta_rho))
        BC_ij = xr.DataArray(BC_ij)
        BC_ij = BC_ij.rename({"dim_0":"along_track","dim_1":"i_j"})
        
        self.BC_lon_lat = BC_lon_lat
        self.BC_ij = BC_ij
        self.smoothed_isobath_coords = smoothed_isobath_coords.copy()
        
        
        return BC_lon_lat, BC_ij
    
    
    def AB_track(self, endpoint_length, number_of_points):
        
        """
        This method gives you track perpendicular to isobath in the southmost point.
        
        Parameter:
        -------------------
            endpoint_length (float): This will set the length of perpendicular track.
            number_of_points (int64): Number of points to get AB divided into.
            
        Results:
        -------------------
            AB_lon_lat (xarray.DataArray): coordinates in lon and lat format.
            AB_ij (xarray.DataArray): coordinates in fractional xi_rhi and eta_rho format.
            
        
        """
        
        self.BC_track()
        
        # coordinates of the isobath (x_isobath, y_isobath)
        x_isobath = self.smoothed_isobath_coords[:,0]
        y_isobath = self.smoothed_isobath_coords[:,1]


        # Choose a point index along the isobath where you want to draw the perpendicular path
        first_point_index1 = 0

        # Calculate the tangent slope along the isobath
        tangent_slope = (y_isobath[first_point_index1 + 1] - y_isobath[first_point_index1]) / (x_isobath[first_point_index1 + 1] - x_isobath[first_point_index1])

        # Calculate the slope of the perpendicular line
        perpendicular_slope = -1 / tangent_slope

        # Define the line equation for the perpendicular path
        x_startpoint, y_startpoint = x_isobath[first_point_index1], y_isobath[first_point_index1]

        line = lambda x: perpendicular_slope * (x - x_startpoint) + y_startpoint

        # Determine the endpoints of the perpendicular path (you can adjust the length)
        x_endpoint = np.array([x_startpoint + endpoint_length])
        y_endpoint = line(x_endpoint)
        
        AB_length = ((x_endpoint - x_startpoint)**2 + (y_endpoint - y_startpoint)**2)**(1/2)
        t = number_of_points
        m = (y_endpoint - y_startpoint)/(x_endpoint - x_startpoint)
        delta = (AB_length*np.cos(np.arctan(m)))/t
        n = np.arange(number_of_points)
        xn = x_startpoint - (n)*delta
        yn = m*(xn - x_startpoint) + y_startpoint
        
        AB_lon_lat = np.column_stack((xn,yn))
        AB_lon_lat = xr.DataArray(AB_lon_lat)
        AB_lon_lat = AB_lon_lat.rename({"dim_0":"along_track","dim_1":"lon_lat"})
        
        
        
        from scipy.interpolate import griddata
        # Replace lat_rho and lon_rho with your actual DataArrays
        lat_rho = self.ds.lat_rho
        lon_rho = self.ds.lon_rho 

        # Replace these with your actual transect coordinates
        transect_lat_rho = AB_lon_lat.values[:,1] 
        transect_lon_rho = AB_lon_lat.values[:,0]

        # Create a 2D meshgrid for xi_rho and eta_rho
        xi_rho  = np.arange(lat_rho.shape[1])
        eta_rho = np.arange(lat_rho.shape[0])
        xi_rho_2d, eta_rho_2d = np.meshgrid(xi_rho, eta_rho)


        # Flatten the 2D arrays
        lat_rho_flat = lat_rho.data.flatten()
        lon_rho_flat = lon_rho.data.flatten()
        xi_rho_flat = xi_rho_2d.flatten()
        eta_rho_flat = eta_rho_2d.flatten()

        # Combine the flattened arrays into a single array
        input_coords = np.column_stack((lat_rho_flat, lon_rho_flat))

        # Interpolate the xi_rho and eta_rho values for the given transect coordinates
        transect_coords = np.column_stack((transect_lat_rho, transect_lon_rho))
        fractional_xi_rho  = griddata(input_coords, xi_rho_flat, transect_coords, method="linear")
        fractional_eta_rho = griddata(input_coords, eta_rho_flat, transect_coords, method="linear")
        
        AB_ij = np.column_stack((fractional_xi_rho, fractional_eta_rho))
        AB_ij = xr.DataArray(AB_ij)
        AB_ij = AB_ij.rename({"dim_0":"along_track","dim_1":"i_j"})
        
        self.AB_lon_lat = AB_lon_lat
        self.AB_ij = AB_ij        
        
        return AB_lon_lat, AB_ij
    
    
    
    
    def CD_track(self, endpoint_length, number_of_points):
        
        
        """
        This method gives you track perpendicular to isobath in the northmost point.
        
        Parameter:
        -------------------
            endpoint_length (float): This will set the length of perpendicular track.
            number_of_points (int64): Number of points to get CD divided into.
            
        Results:
        -------------------
            CD_lon_lat (xarray.DataArray): coordinates in lon and lat format.
            CD_ij (xarray.DataArray): coordinates in fractional xi_rhi and eta_rho format.
            
        
        """
        
        self.BC_track()
        
        # coordinates of the isobath (x_isobath, y_isobath)
        x_isobath = self.smoothed_isobath_coords[:,0]
        y_isobath = self.smoothed_isobath_coords[:,1]


        # Choose a point index along the isobath where you want to draw the perpendicular path
        end_point_index1 = -1

        # Calculate the tangent slope along the isobath
        tangent_slope = (y_isobath[end_point_index1] - y_isobath[end_point_index1-1]) / (x_isobath[end_point_index1] - x_isobath[end_point_index1-1])

        # Calculate the slope of the perpendicular line
        perpendicular_slope = -1 / tangent_slope

        # Define the line equation for the perpendicular path
        x_startpoint, y_startpoint = x_isobath[end_point_index1], y_isobath[end_point_index1]

        line = lambda x: perpendicular_slope * (x - x_startpoint) + y_startpoint

        # Determine the endpoints of the perpendicular path (you can adjust the length)
        x_endpoint = np.array([x_startpoint + endpoint_length])
        y_endpoint = line(x_endpoint)
        
        CD_length = ((x_endpoint - x_startpoint)**2 + (y_endpoint - y_startpoint)**2)**(1/2)
        t = number_of_points
        m = (y_endpoint - y_startpoint)/(x_endpoint - x_startpoint)
        delta = (CD_length*np.cos(np.arctan(m)))/t
        n = np.arange(number_of_points)
        xn = x_startpoint - n*delta
        yn = m*(xn - x_startpoint) + y_startpoint
        
        CD_lon_lat = np.column_stack((xn,yn))
        CD_lon_lat = xr.DataArray(CD_lon_lat)
        CD_lon_lat = CD_lon_lat.rename({"dim_0":"along_track","dim_1":"lon_lat"})
        
        
        
        from scipy.interpolate import griddata
        # Replace lat_rho and lon_rho with your actual DataArrays
        lat_rho = self.ds.lat_rho
        lon_rho = self.ds.lon_rho 

        # Replace these with your actual transect coordinates
        transect_lat_rho = CD_lon_lat.values[:,1] 
        transect_lon_rho = CD_lon_lat.values[:,0]

        # Create a 2D meshgrid for xi_rho and eta_rho
        xi_rho  = np.arange(lat_rho.shape[1])
        eta_rho = np.arange(lat_rho.shape[0])
        xi_rho_2d, eta_rho_2d = np.meshgrid(xi_rho, eta_rho)


        # Flatten the 2D arrays
        lat_rho_flat = lat_rho.data.flatten()
        lon_rho_flat = lon_rho.data.flatten()
        xi_rho_flat = xi_rho_2d.flatten()
        eta_rho_flat = eta_rho_2d.flatten()

        # Combine the flattened arrays into a single array
        input_coords = np.column_stack((lat_rho_flat, lon_rho_flat))

        # Interpolate the xi_rho and eta_rho values for the given transect coordinates
        transect_coords = np.column_stack((transect_lat_rho, transect_lon_rho))
        fractional_xi_rho  = griddata(input_coords, xi_rho_flat, transect_coords, method="linear")
        fractional_eta_rho = griddata(input_coords, eta_rho_flat, transect_coords, method="linear")
        
        CD_ij = np.column_stack((fractional_xi_rho, fractional_eta_rho))
        CD_ij = xr.DataArray(CD_ij)
        CD_ij = CD_ij.rename({"dim_0":"along_track","dim_1":"i_j"})
        
        self.CD_lon_lat = CD_lon_lat
        self.CD_ij = CD_ij        
        
        
        return CD_lon_lat, CD_ij
    
    
    def plot_map(self, lon_east, lon_west, lat_north, lat_south, AB_lon_lat, BC_lon_lat, CD_lon_lat):
        
        projection = ccrs.PlateCarree()
        
        fig, ax = plt.subplots(figsize = (10,6), subplot_kw = {"projection" : projection})

        # Add coastlines and land features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor=cfeature.COLORS['land'])

        # Set the extent of the map (optional, based on your area of interest)
        ax.set_extent([lon_west, lon_east, lat_south, lat_north])

        # Add gridlines with labels
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = ccrs.cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = ccrs.cartopy.mpl.gridliner.LATITUDE_FORMATTER

        # Plot the isobath, coastline, and perpendicular path
        plt.plot(BC_lon_lat[:,0], BC_lon_lat[:,1], color="red", label="Isobath")
        plt.plot(AB_lon_lat[:,0], AB_lon_lat[:,1], color="blue", label="perpendicular path")
        plt.plot(CD_lon_lat[:,1], CD_lon_lat[:,1], color="green", label="perpendicular path")
        # plt.scatter(x_start1, y_start1, color="red", marker="o", label="Selected Point")
        # plt.scatter(x_start2, y_start2, color="red", marker="o", label="Selected Point")

        plt.legend()

        plt.savefig('./isobath.png')
        plt.show()
        
        return fig
    
    def checkplot(self, lon_east, lon_west, lat_north, lat_south, **kwargs):
        """
        This method is to check the desired isobath track for the analysis before finalizing the coordinates to interpolate variables on.
        
        Arguments:
            - lon_east (float): Eastern boundary longitude
            - lon_west (float): Western boundary longitude
            - lat_north (float): Northern boundary latitude
            - lat_south (float): Southern boundary latitude
            - **kwargs: additional keyword argument for specifying figure size by passing a tuple to fig_size variable and AB_endpoint and CD_endpoint for the length of AB and CD track respectively.
        
        
        Example:
            >>> instance = track.checkplot(lon_east, lon_west, lat_north, lat_south, fig_size = (10,6), AB_endpoint=-0.74, CD_endpoint=0.32)
        
        
        Result:
            None: Displays a plot showing the track along specified isobath and perpendicular track closing it.
        
        
        
        """
        AB_endpoint = kwargs.get('AB_endpoint', -0.75)
        number_of_points = 50
        CD_endpoint = kwargs.get('CD_endpoint', 0.75)
        self.BC_track()
        self.AB_track(AB_endpoint, number_of_points)
        self.CD_track(CD_endpoint, number_of_points)
        
        
        projection = ccrs.PlateCarree()
        
        figsize = kwargs.get('fig_size', (10,6))
        
        fig, ax = plt.subplots(figsize = figsize, subplot_kw = {"projection": projection})
        
        # Add coastlines and land features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor=cfeature.COLORS['land'])

        # Set the extent of the map (optional, based on your area of interest)
        ax.set_extent([lon_west, lon_east, lat_south, lat_north])

        # Add gridlines with labels
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = ccrs.cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = ccrs.cartopy.mpl.gridliner.LATITUDE_FORMATTER

        # Plot the isobath, coastline, and perpendicular path
        plt.plot(self.BC_lon_lat[:,0], self.BC_lon_lat[:,1], color="red", label="Isobath BC")
        plt.plot(self.AB_lon_lat[:,0], self.AB_lon_lat[:,1], color="blue", label="perpendicular path AB")
        plt.plot(self.CD_lon_lat[:,0], self.CD_lon_lat[:,1], color="green", label="perpendicular path CD")
        # plt.scatter(x_start1, y_start1, color="red", marker="o", label="Selected Point")
        # plt.scatter(x_start2, y_start2, color="red", marker="o", label="Selected Point")

        plt.legend()

        plt.savefig('./isobath.png')
        plt.show()        

    
    
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        