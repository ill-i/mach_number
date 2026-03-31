import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

import csv

import pandas as pd

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

import os
import scipy
from scipy.optimize import curve_fit
from scipy.stats import norm


def load_and_combine_data(csv_file1, csv_file2, field_num):
    """
    Loads data from two CSV files and merges them into one DataFrame.

    Arguments:
    csv_file1 (str): Path to the first CSV file.
    csv_file2 (str): Path to the second CSV file.
    field_num (int)

    Returns:
    tuple: a tuple of three DataFrame objects:
        - Merged DataFrame from two files.
        - DataFrame loaded from the first file.
        - DataFrame loaded from the second file.
    """
    cat1 = pd.read_csv(csv_file1, delim_whitespace=True)
    cat2 = pd.read_csv(csv_file2, delim_whitespace=True)
    frames = [cat1, cat2]
    cat = pd.concat(frames)
    cat['field'] = int(field_num) 
    
    return cat,cat1,cat2


def process_fits_file(imaging_file):
    """
    Opens a FITS file and prepares a WCS object based on its header.
    
    Arguments:
    imaging_file (str): Path to the FITS file.
    
    Returns:
    tuple: A tuple containing:
        - HDUList object from the FITS file.
        - WCS object created from the header of the primary HDU.
    """
    # Open the FITS file
    hdu = fits.open(imaging_file)
    # Extract the header from the primary HDU
    header = hdu[0].header
    # Create a WCS object from the header
    wcs = WCS(header)
    
    return hdu, wcs

def calculate_coordinates(cat, wcs, shift_x,shift_y):
    """
    Calculates astronomical coordinates from pixel coordinates with a shift applied to X coordinates.

    Arguments:
    - cat: DataFrame containing columns 'X_IMAGE' and 'Y_IMAGE' with pixel coordinates.
    - wcs: WCS object for converting pixel coordinates to sky coordinates.
    - shift: Numeric value to be added to 'X_IMAGE' coordinates before conversion.

    Returns:
    - List of tuples, where each tuple contains (RA, DEC) in degrees.
    """
    # Apply the shift to the X coordinates and keep Y coordinates as they are
    x_image_shifted = cat['X_IMAGE'] + shift_x
    y_image = cat['Y_IMAGE'] + shift_y

    # Convert the pixel coordinates to world coordinates (RA, DEC)
    ra, dec = wcs.wcs_pix2world(x_image_shifted, y_image, 1)

    # Return the coordinates as a list of (RA, DEC) tuples
    return list(zip(ra, dec))

def magfunction(x, A, B):
    """
    Calculate the magnitude function for an array of x values.

    Arguments:
    - x: A scalar or an array of values.
    - A, B: Parameters of the magnitude function.

    Returns:
    - y: The calculated magnitude values as a scalar or an array.
    """
    y = A - B * np.log10(x)
    return y

def first_neighbors(offset_radius,cat,gda,coords):
    """
    Finds the nearest Gaia catalog entry for each entry in the 'cat' catalog within a specified 'offset_radius'.
    (first interaction)
    Parameters:
    - offset_radius: The radius within which to search for neighbors.
    - cat: The catalog of interest (Kanata).
    - gda: The Gaia catalog, expected to be a pandas DataFrame.
    - coords: A list of [RA, DEC] coordinates for each entry in 'cat'.
    
    Returns:
    - nearby_arr: Distances to the nearest star in the Gaia catalog.
    - inds_gaia: Indices in the Gaia catalog of the nearest neighbors.
    - inds_cat: Indices in the 'cat' catalog.
    """

    nn = len(cat)
    nb = len(gda)
    
    nearby_arr = np.full(nn, np.nan)  # Distance to the nearest star
    inds_gaia  = np.full(nn, np.nan)    # Index in the gaia catalog
    inds_cat   = np.full(nn, np.nan)    # Index in the kanata catalog
    
    orig_stars = np.zeros(nn)         # Control the indices of kanata catalog data
    
    # Extract RA and DEC from gda
    ra_gda  = gda.iloc[:, 5].values #if gda like pandas DF
    dec_gda = gda.iloc[:, 7].values
   
    for i in range(nn):
        
        dra = ra_gda - coords[i][0]
        ddec = dec_gda - coords[i][1]
        
        distarr = np.sqrt(dra**2 + ddec**2)

        nb_nearby = []
        nearby_inds = []
        
        for j in range(nb-1):
            if distarr[j] <= offset_radius:
                nb_nearby.append(distarr[j])
                nearby_inds.append(j)
        
        nl = len(nearby_inds)
        
        if nl == 0:
            nearby_arr[i] = np.nan
            inds_gaia[i]  = np.nan 
            inds_cat[i]   = np.nan
            orig_stars[i] = 0
            
        elif nl == 1:
            nearby_arr[i] = nb_nearby[0]
            inds_gaia[i]  = nearby_inds[0]
            inds_cat[i]   = i
            orig_stars[i] = 1
            
        elif nl > 1:
            imin = np.argmin(nb_nearby)
            nearby_arr[i] = nb_nearby[imin]
            inds_gaia[i]  = nearby_inds[imin]
            inds_cat[i]   = i
            orig_stars[i] = nl

    return nearby_arr, inds_gaia, inds_cat


def magvsflux(inds_gaia, inds_cat, flux_cat, hmag_gaia):
    """
    Analyzes and visualizes the relationship between flux and magnitude of stars,
    using data from Gaia and another catalog.
    """
    # Filtering data to exclude NaN values
    mask_gaia = ~np.isnan(inds_gaia)
    mask_cat = ~np.isnan(inds_cat)
    
    # Check that the filtered data are positive and correspond to each other
    if np.any(flux_cat[inds_cat[mask_cat].astype(int)] <= 0):
        raise ValueError("All flux values must be positive.")
    
    filtered_flux_cat = flux_cat[inds_cat[mask_cat].astype(int)]
    filtered_hmag_gaia = hmag_gaia[inds_gaia[mask_gaia].astype(int)]
    
    try:
        # Fitting the model
        parameters, covariance = curve_fit(magfunction, filtered_flux_cat, filtered_hmag_gaia)
    except RuntimeError as e:
        print("Model fitting error:", e)
        return None
    
    # Visualizing data and the model
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_flux_cat, filtered_hmag_gaia, color='red', label='Observed Data')
    plt.plot(filtered_flux_cat, magfunction(filtered_flux_cat, *parameters), 'b.', label='Model')
    plt.xscale('log')
    plt.xlabel('Flux (from Kanata)')
    plt.ylabel('Magnitude (from Gaia)')
    # plt.title('offset raduis = 0.0005')
    plt.legend()
    # plt.show()
#plt.savefig(figdir + 'v1_g111-6_HbandPOL1.png', format=None, transparent=False, bbox_inches=None, pad_inches=0.1)
    # plt.savefig('magvsflux_8.png', dpi = 150)
    print("Model parameters:", parameters)
    return parameters


def second_neighbors(offset_radius,cat,gda,coords,parameters,flux_cat,hmag_gaia):
    """
    Finds and refines nearest neighbors within a given offset radius based on spatial proximity and photometric similarity.
      (second interaction)
    Parameters:
    - offset_radius: Radius within which to search for neighbors.
    - cat: The catalog of interest (Kanata).
    - gda: Gaia catalog as a pandas DataFrame.
    - coords: List of [RA, DEC] coordinates for objects in `cat`.
    - parameters: [A, B] parameters from a magnitude-flux relationship.
    - flux_cat: Observed fluxes in `cat`.
    - hmag_gaia: Gaia magnitudes corresponding to `gda`.
    
    Returns:
    - nearby_arr: Distances to nearest stars in Gaia catalog.
    - inds_gaia: Indices of matching stars in Gaia catalog.
    - inds_cat: Indices in the input catalog.
    """
    
    nn = len(cat)
    nb = len(gda)
    
    nearby_arr = np.full(nn, np.nan)  # Distance to the nearest star

    inds_gaia  = np.full(nn, np.nan)    # Index in the gaia catalog
   
    inds_cat   = np.full(nn, np.nan)    # Index in the kanata catalog

    orig_stars = np.zeros(nn)   ## to control the indices of kanata catalog data

     # Extract RA and DEC from gda
    ra_gda  = gda.iloc[:, 5].values #if gda like pandas DF
    dec_gda = gda.iloc[:, 7].values
   
    for i in range(nn):
        
        dra = ra_gda - coords[i][0]
        ddec = dec_gda - coords[i][1]
        
        distarr = np.sqrt(dra**2 + ddec**2)

        nb_nearby = []
        nearby_inds = []
        
        for j in range(nb-1):
            if distarr[j] <= offset_radius:
                nb_nearby.append(distarr[j])
                nearby_inds.append(j)
                
        nl = len(nearby_inds)
        if nl == 0:
            nearby_arr[i] = np.nan
            inds_gaia[i]  = np.nan 
            inds_cat[i]   = np.nan
            orig_stars[i] = 0
            
        elif nl == 1:
            nearby_arr[i] = nb_nearby[0]
            inds_gaia[i]  = nearby_inds[0]
            inds_cat[i]   = i
            orig_stars[i] = 1
            
        if nl > 1:
             # Calculate the expected magnitude from the observed flux
            mag_emp = parameters[0] - parameters[1] * np.log10(flux_cat[i])
            mag_diff = hmag_gaia[nearby_inds] - mag_emp 
            
            # Select the star with the smallest magnitude difference
            min_diff_index = np.argmin(np.abs(mag_diff))
            nearby_arr[i] = nb_nearby[min_diff_index]
            inds_gaia[i] = nearby_inds[min_diff_index]
            inds_cat[i] = i
            
    return nearby_arr, inds_gaia, inds_cat

def cross_match_Kanata_Gaia(cat, wcs, gaia, shift_x,shift_y, offset_radius = 10./3600):
    """
    Cross-matches objects from a Kanata catalog with the Gaia catalog based on spatial coordinates and refines matches using photometric data.

    Parameters:s
    - cat: Pandas DataFrame containing the Kanata catalog data.
    - wcs: WCS object representing the World Coordinate System of the Kanata observations.
    - gaia: Pandas DataFrame containing the Gaia catalog data.
    - shift: Tuple or list indicating the shift in coordinates to apply to the Kanata catalog before matching.
    - offset_radius: The radius within which to search for Gaia neighbors, in degrees (default is 10 arcseconds).

    Returns:
    - matched_cat: Pandas DataFrame of the Kanata catalog with Gaia matches and additional Gaia data merged in.
    - inds_gaia: Array of Gaia index values corresponding to the Kanata catalog entries.
    """
    
    
    flux_cat = np.array(cat['FLUX_ALL'])
    hmag_gaia = gaia['Hmag']
    coords = calculate_coordinates(cat, wcs, shift_x,shift_y)
    nearby_arr_1, inds_gaia_1, inds_cat_1 = first_neighbors(offset_radius, cat, gaia, coords)
    parameters = magvsflux(inds_gaia_1, inds_cat_1, flux_cat, hmag_gaia)

    nearby_arr_2, inds_gaia_2, inds_cat_2 = second_neighbors(offset_radius,cat,gaia,coords,parameters,flux_cat,hmag_gaia)
    
    inds_gaia = np.nan_to_num(inds_gaia_2, nan=-1).astype(int)

    # Create a DataFrame to store Gaia indexes
    gaia_indices = pd.DataFrame({'Gaia_index': inds_gaia})
    gaia_indices = gaia_indices[gaia_indices['Gaia_index'] >= 0]  # Фильтруем отсутствующие совпадения

    # Add indexes to the original DataFrame cat
    cat_with_gaia_indices = cat.copy()
    cat_with_gaia_indices = cat_with_gaia_indices.reset_index()
    cat_with_gaia_indices['Gaia_index'] = gaia_indices['Gaia_index'].reset_index(drop=True)

    # Combine cat with information from Gaia
    matched_cat = pd.merge(cat_with_gaia_indices, gaia, left_on='Gaia_index', right_index=True, how='left')

    return  matched_cat, nearby_arr_2


#exmpl

def classify_distance(distance):
    if distance < 2.7*10**3:
        return 'less than 2.7kpc'
    elif 2.7*10**3 <= distance < 3.2*10**3:
        return 'between 2.7kpc and 3.2kpc'
    elif 3.2*10**3 <= distance:
        return 'greater than 3.2kpc'
    else:
        return 'no distance'

    

def classify_distance_1660(distance):
    if distance < 1.660*10**3:
        return 'less than 1.66kpc'
    elif 1.660**3 <= distance < 3.2*10**3:
        return 'between 1.66kpc and 3.2kpc'
    elif 3.2*10**3 <= distance:
        return 'greater than 3.2kpc'
    else:
        return 'no distance'
    
    
def minimal_angle_between_lines(angle1, angle2):
    """
    Calculates the minimum angle between two lines.
     """
     # Normalize angle difference to range [0, 180]
    delta_angle = abs(angle1 - angle2) % 360
    if delta_angle > 180:
        delta_angle = 360 - delta_angle
        

    return min(delta_angle, 180 - delta_angle)


