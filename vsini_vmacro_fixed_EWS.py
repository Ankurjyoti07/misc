import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
import glob, warnings, gzip, logging, os, sys, csv, re
from math import sin, pi
from scipy.special import erf                               # Error function 
from lmfit import Parameters, minimize, report_fit
from lmfit.confidence import conf_interval
from math import ceil, sqrt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from astropy import constants, units
import sys, glob, os, logging, warnings, csv
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as svg
from astropy.constants import c
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks, fftconvolve
from astropy.stats import sigma_clip
from tqdm import tqdm
from scipy.interpolate import griddata

def sort_spectra(spectra_files):
    spectra_with_dates = []
    for spectrum_file in spectra_files:
        obs_date = fits.getheader(spectrum_file)['MJD-OBS']
        if obs_date:
            spectra_with_dates.append((spectrum_file, obs_date))
    spectra_with_dates.sort(key=lambda x: x[1])
    sorted_spectra_files = [s[0] for s in spectra_with_dates]
    sorted_obs_dates = [s[1] for s in spectra_with_dates]
    return sorted_spectra_files, sorted_obs_dates


def read_ADP(infile_adp):
    data = fits.getdata(infile_adp)
    wave , flux = data[0], data[2]
    return wave, flux

def read_spectrum(infile):
    data = fits.getdata(infile)
    wave , flux = data[0], data[1]
    return wave, flux

def get_mean_flux(glob_specdir):

    all_flux = []
    
    for spectrum_file in glob_specdir:
        wavelengths, flux = read_ADP(spectrum_file)
        all_flux.append(flux)
    
    all_flux = np.array(all_flux)
    mean_flux = np.nanmedian(all_flux, axis=0)
    return wavelengths , mean_flux

def get_mean_flux_manual(glob_specdir):

    all_flux = []
    
    for spectrum_file in glob_specdir:
        wavelengths, flux = read_spectrum(spectrum_file)
        all_flux.append(flux)
    
    all_flux = np.array(all_flux)
    mean_flux = np.nanmedian(all_flux, axis=0)
    return wavelengths , mean_flux

def telluric_correction(wv, flx):
    window_size = 100
    cleaned_flx = np.copy(flx)

    for i in range(0, len(wv), window_size):
        flx_window = flx[i:i + window_size]
        clipped_flux = sigma_clip(flx_window, sigma=2, maxiters=10, masked=True)
        cleaned_flx[i:i + window_size] = np.where(clipped_flux.mask, np.nan, flx_window)

    cleaned_flx = np.interp(wv, wv[~np.isnan(cleaned_flx)], cleaned_flx[~np.isnan(cleaned_flx)])
    return wv, cleaned_flx

def read_line_list(filename):
    line_centers = []
    line_widths = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  #removing whitespaces
            if not line:
                continue  #skipping empty lines
            parts = line.split()
            center = float(parts[0])
            if len(parts) > 1:
                width = float(parts[1])
            else:
                width = 10.0  #default
            line_centers.append(center)
            line_widths.append(width)

    return line_centers, line_widths

class Model_broad:
    def __init__(self, wave, flux):
        self.x = wave
        self.y = flux


def Broaden(model, vsini, epsilon=0.5, linear=False, findcont=False):
    # Remove NaN values from the flux array and corresponding wavelength values
    non_nan_idx = ~np.isnan(model.y)
    wvl = model.x[non_nan_idx]
    flx = model.y[non_nan_idx]
    
    dwl = wvl[1] - wvl[0]
    binnu = int(np.floor((((vsini/10)/ 299792.458) * max(wvl)) / dwl)) + 1 #adding extra bins for error handling
    #validIndices = np.arange(len(flx)) + binnu => this was used in rotbroad as a user cond ==> this is always on here
    front_fl = np.ones(binnu) * flx[0]
    end_fl = np.ones(binnu) * flx[-1]
    flux = np.concatenate((front_fl, flx, end_fl))

    front_wv = (wvl[0] - (np.arange(binnu) + 1) * dwl)[::-1]
    end_wv = wvl[-1] + (np.arange(binnu) + 1) * dwl
    wave = np.concatenate((front_wv, wvl, end_wv))

    if not linear:
        x = np.logspace(np.log10(wave[0]), np.log10(wave[-1]), len(wave))
    else:
        x = wave
        
    if findcont:
        # Find the continuum
        model.cont = np.ones_like(flux)  # Placeholder for continuum finding
        
    # Make the broadening kernel
    dx = np.log(x[1] / x[0])
    c = 299792458  # Speed of light in m/s
    lim = vsini / c
    if lim < dx:
        warnings.warn("vsini too small ({}). Not broadening!".format(vsini))
        return Model_broad(wave.copy(), flux.copy())  # Create a copy of the Model object
    
    d_logx = np.arange(0.0, lim, dx)
    d_logx = np.concatenate((-d_logx[::-1][:-1], d_logx))
    alpha = 1.0 - (d_logx / lim) ** 2
    B = (1.0 - epsilon) * np.sqrt(alpha) + epsilon * np.pi * alpha / 4.0  # Broadening kernel
    B /= np.sum(B)  # Normalize

    # Do the convolution
    broadened = Model_broad(wave.copy(), flux.copy())  # Create a copy of the Model object
    broadened.y = fftconvolve(flux, B, mode='same')
    
    return broadened

def macro_broaden(xdata, ydata, vmacro):
    c = 299792458 #~constants.c.cgs.value * units.cm.to(units.km)
    sq_pi = np.sqrt(np.pi)
    lambda0 = np.median(xdata)
    xspacing = xdata[1] - xdata[0]
    mr = vmacro * lambda0 / c
    ccr = 2 / (sq_pi * mr)

    px = np.arange(-len(xdata) / 2, len(xdata) / 2 + 1) * xspacing
    pxmr = abs(px) / mr
    profile = ccr * (np.exp(-pxmr ** 2) + sq_pi * pxmr * (erf(pxmr) - 1.0))

    before = ydata[int(-profile.size / 2 + 1):]
    after = ydata[:int(profile.size / 2 +1)] #add one to fix size mismatch
    extended = np.r_[before, ydata, after]

    first = xdata[0] - float(int(profile.size / 2.0 + 0.5)) * xspacing
    last = xdata[-1] + float(int(profile.size / 2.0 + 0.5)) * xspacing
    
    x2 = np.linspace(first, last, extended.size)  #newdata x array ==> handles edge effects

    conv_mode = "valid"

    newydata = fftconvolve(extended, profile / profile.sum(), mode=conv_mode)

    return newydata


def compute_equivalent_width(wavelength, flux, lambda_min, lambda_max):
    mask = (wavelength >= lambda_min) & (wavelength <= lambda_max)
    wv_line = wavelength[mask]
    flx_line = flux[mask]
    ew = np.trapz(1 - flx_line, wv_line)    
    return ew

def compute_ews_from_linelist(wavelength, flux, linelist_file):
    line_centers, line_widths = read_line_list(linelist_file)
    
    ews = []
    for center, width in zip(line_centers, line_widths):
        lambda_min = center - (width / 2)
        lambda_max = center + (width / 2)
        ew = compute_equivalent_width(wavelength, flux, lambda_min, lambda_max)
        ews.append(ew)
    return ews


def gauss_EW(x,center,R, EW, gamma):
  a = EW*R/(1.064*center)
  sigma = center/ (2.0 * R * np.sqrt(2.0 * np.log(2))) 
  return -a*np.exp(-(x-center)**2/(2*sigma**2)) + gamma

def generate_data(wave, flux, line_centers, line_widths, wavelength_slices):
    interp_func = interp1d(wave, flux, kind='linear')
    wave_slices = []
    flux_slices = []
    for center, width in zip(line_centers, line_widths):
        new_wave = np.linspace(center - width, center + width, wavelength_slices)
        new_flux = interp_func(new_wave)
        wave_slices.append(new_wave)
        flux_slices.append(new_flux)
    return np.concatenate(wave_slices), np.concatenate(flux_slices)


def generate_broaden(params, line_centers, line_widths, wavelength_slices, ew, vsini = 400000):
    model_slices = []
    for i, (center, width) in enumerate(zip(line_centers, line_widths)):
        wave = np.linspace(center - width, center + width, wavelength_slices)
        
        instrum = gauss_EW(wave, params[f'center{i}'], 180000, ew[i], 1) #params[f'gamma{i}']///resolution is still hardcoded R=20000 change accordingly
        broad_rot = Broaden(Model_broad(wave, instrum), vsini)
        broad_macro = macro_broaden(broad_rot.x, broad_rot.y, params[f'vmacro{i}']) #macro broad restores the same wave array as input  
        interp = interp1d(broad_rot.x, broad_macro, kind= 'linear')
        broad_flux = interp(wave)
        model_slices.append(broad_flux)
    return  np.concatenate(model_slices)

def objective(params, wave, flux, line_centers, line_widths, wavelength_slices, ew, vrot = 400000):
    wave_data, flux_data = generate_data(wave, flux, line_centers, line_widths, wavelength_slices)
    model = generate_broaden(params, line_centers, line_widths, wavelength_slices, ew, vsini = vrot)
    return flux_data - model

def fit_lines(wave, flux, line_centers, line_widths, wavelength_slices, ew, vrot):
    params = Parameters()
    wave_data, flux_data = generate_data(wave, flux, line_centers, line_widths, wavelength_slices)
    for i, (center, width) in enumerate(zip(line_centers, line_widths)):
        params.add(f'center{i}', value=center, min = center-1, max = center+1)  # Initial guess for center
        #params.add(f'gamma{i}', value=1, min = 0.5, max = 1.2)
        params.add(f'vmacro{i}', value=50000, min = 20000, max = 600000)
    result = minimize(objective, params=params, args=(wave_data, flux_data, line_centers, line_widths, wavelength_slices, ew, vrot))
    return result

def diagnostic_plots(result, wave, flux, line_centers, line_widths, ews, wavelength_slices=1000, vrot=330000, specfile='plot.png', save_dir='../checks/full_linelist_ews_vmacro_o2/'):
    num_lines = len(line_centers)
    cols = int(np.ceil(np.sqrt(num_lines)))  # Number of columns
    rows = int(np.ceil(num_lines / cols))    # Number of rows

    fig = plt.figure(figsize=(cols * 5, rows * 4))  

    #spacing
    dx = 0.85 / cols 
    dy = 0.8 / rows  
    gap_between_pairs = 0.09
    fit_height = (dy - gap_between_pairs) * 0.7
    residual_height = (dy - gap_between_pairs) * 0.3

    for i, (center, width) in enumerate(zip(line_centers, line_widths)):
        row = i // cols
        col = i % cols
        x_start = 0.07 + col * (dx + gap_between_pairs)
        y_start = 0.1 + (rows - row - 1) * (dy + gap_between_pairs)
        line_wave = np.linspace(center - width, center + width, wavelength_slices)
        model_params = result.params
        line_center = model_params[f'center{i}'].value
        gamma = 1 #model_params[f'gamma{i}'].value
        vmacro = model_params[f'vmacro{i}'].value
        vsini = vrot

        instrum = gauss_EW(line_wave, line_center, 180000, ews[i], gamma)
        broadened_model = Broaden(Model_broad(line_wave, instrum), vsini)
        broadened_flux = macro_broaden(broadened_model.x, broadened_model.y, vmacro)
        interp_flux = interp1d(wave, flux, kind='linear')(line_wave)
        model_interp = interp1d(broadened_model.x, broadened_flux, kind='linear')(line_wave)

        ax_fit = fig.add_axes([x_start, y_start + residual_height, dx, fit_height])
        ax_fit.plot(line_wave, interp_flux, label='Data', color='black', alpha = 0.7)
        ax_fit.plot(line_wave, model_interp, label='Model', color='red', linestyle='--')
        ax_fit.set_title(f"{center} Å | vmacro: {vmacro / 1000:.3f} km/s", fontsize=14)
        ax_fit.set_ylabel("Flux", fontsize=14)
        ax_fit.legend(fontsize=8)
        ax_fit.tick_params(axis='both', which='both', labelsize=14)  # Bigger tick labels
        ax_fit.set_xlabel("Wavelength (Å)", fontsize=14)

        ax_residual = fig.add_axes([x_start, y_start, dx, residual_height], sharex=ax_fit)
        residuals = interp_flux - model_interp
        ax_residual.plot(line_wave, residuals, '.r')
        ax_residual.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax_residual.set_xlabel("Wavelength (Å)", fontsize=14)
        ax_residual.set_ylabel("Residuals", fontsize=14)
        ax_residual.tick_params(axis='both', which='both', labelsize=14)
        
    #plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, specfile)
    plt.savefig(file_path)
    plt.show()
    plt.close()


def process_spectra(folder, linelist_file='line_list.txt', output_file='vmacro_results_renorm_o2.txt'):
    line_centers, line_widths = read_line_list(linelist_file)
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as f:
            header = "Filename\t" + "\t".join([f"vmacro_{i+1}" for i in range(4)]) + "\t" + "\t".join([f"error_{i+1}" for i in range(4)]) + "\n"
            f.write(header)

    spectra_files = glob.glob(folder)

    for spectrum in tqdm(spectra_files, desc="Processing Spectra"):
        try:
            spectrum_name = os.path.basename(spectrum)
            wave, flux = read_spectrum(spectrum)
            wave, flux = telluric_correction(wave, flux)
            ews = compute_ews_from_linelist(wave, flux, linelist_file)
            vrot = 340000
            result = fit_lines(wave, flux, line_centers, line_widths, wavelength_slices=1000, ew=ews, vrot=vrot)
            result
            diagnostic_plots(result, wave, flux, line_centers, line_widths, ews=ews, wavelength_slices=1000, specfile=spectrum_name+'.png')
        except Exception as e:
            print(f"Error processing {spectrum_name}: {e}")

'''
usage:  for single spectrum, use fit_lines and diagnostic plots function with their variables : 
        for eg:  line_centers, line_widths = read_line_list('line_list.txt')
                ews = compute_ews_from_linelist(wave, flux, 'line_list.txt')
                result = fit_lines(wave, flux, line_centers, line_widths, wavelength_slices=1000, ew=ews, vrot=vrot)
                dignostic_plots(result, wave, flux, line_centers, line_widths, ews=ews)
        for bulk usage, use the proces_spectra function and specify the folders containing the spectra and the folders where you want to store the outputs
'''