import glob, math, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.io import fits


def read_spectrum(infile):
    with fits.open(infile) as hdul:
        wave, norm_flux = hdul[1].data, hdul[2].data
    return wave, norm_flux

def read_line_list(filename):
    line_centers = []
    line_widths = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            center = float(parts[0])
            if len(parts) > 1:
                width = float(parts[1])
            else:
                width = 13.0
            line_centers.append(center)
            line_widths.append(width)

    return line_centers, line_widths

def sort_spectra(spectra_files):
    spectra_with_dates = []
    for spectrum_file in spectra_files:
        obs_date = fits.getheader(spectrum_file)['MJD-OBS']
        if obs_date:
            spectra_with_dates.append((spectrum_file, obs_date))
    spectra_with_dates.sort(key=lambda x: x[1])
    sorted_spectra_files = [s[0] for s in spectra_with_dates]
    return sorted_spectra_files

def compute_residuals(spectra_files):
    """
    Compute residuals by subtracting the mean spectrum from each individual spectrum.
    
    Parameters:
    - spectra_files: list of spectrum files.
    
    Returns:
    - wavelengths: array of wavelengths.
    - residuals: 2D array of residuals (each row corresponds to residuals for one spectrum).
    - observation_dates: array of observation dates.
    """
    all_flux = []
    observation_dates = []

    for spectrum_file in spectra_files:
        wavelengths, flux = read_spectrum(spectrum_file)
        all_flux.append(flux)
        obs_date = fits.getheader(spectrum_file)['MJD-OBS']
        observation_dates.append(obs_date)

    all_flux = np.array(all_flux)
    observation_dates = np.array(observation_dates)
    mean_flux = np.mean(all_flux, axis=0)
    residuals = all_flux - mean_flux
    return wavelengths, residuals, observation_dates

def plot_time_series_residuals_by_lines(spectra_files, line_centers, line_widths):
    """
    Plot the time-series residuals for specific wavelength ranges from the linelist.
    Each line will be in its own subplot.

    Parameters:
    - spectra_files: list of spectrum files.
    - line_centers: list of line centers from the linelist.
    - line_widths: list of line widths from the linelist.
    """

    wavelengths, residuals, observation_dates = compute_residuals(spectra_files)

    num_lines = len(line_centers)
    grid_size = math.ceil(math.sqrt(num_lines))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 10))
    axes = axes.flatten()

    for i, (center, width) in enumerate(zip(line_centers, line_widths)):
        ax = axes[i]
        
        line_mask = (wavelengths >= center - width) & (wavelengths <= center + width)
        selected_wavelengths = wavelengths[line_mask]
        selected_residuals = residuals[:, line_mask]
        mesh = ax.pcolormesh(selected_wavelengths, observation_dates, selected_residuals, shading='auto', cmap='RdBu_r')
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label('Residual Flux')

        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Observation Date (MJD)')
        ax.set_title(f'Line {center:.1f} Å')

    for j in range(i + 1, grid_size * grid_size):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

spectra_files = glob.glob('/home/c4011027/PhD_stuff/ESO_proposals/zetaOph_norm/*.fits')
sorted_spectra = sort_spectra(spectra_files)[1400:1550]
line_centers, line_widths = read_line_list('/home/c4011027/PhD_stuff/ESO_proposals/prologs/line_list.txt')
plot_time_series_residuals_by_lines(sorted_spectra, line_centers, line_widths)