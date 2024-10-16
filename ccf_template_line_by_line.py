import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import scipy.optimize as so
import os, csv, glob
from astropy.io import fits
from tqdm import tqdm

def gaussian(x, center, height, std, yoffset):
    return height * np.exp(-1 * (x - center)**2 / (2*std**2)) + yoffset

def double_gaussian(x, center1, height1, std1, center2, height2, std2, yoffset):
    return gaussian(x, center1, height1, std1, 0) + gaussian(x, center2, height2, std2, 0) + yoffset

def quadratic(x, q, a, c):
    b = -2*q*a
    return a * x**2 + b*x + c

def dopler_shift(w, rv):
    w = np.array(w)
    c = 299792.458
    return w * c / (c - rv)


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
                width = 14.0
            line_centers.append(center)
            line_widths.append(width)

    return line_centers, line_widths

def compute_mean_flux_for_lines(spectra_files, line_centers, line_widths):
    all_flux = []
    for spectrum_file in spectra_files:
        wavelengths, flux = read_spectrum(spectrum_file)
        all_flux.append(flux)    
    all_flux = np.array(all_flux)
    mean_flux = np.mean(all_flux, axis=0)

    mean_flux_for_lines = []
    for i, center in enumerate(line_centers):
        lower_bound = center - line_widths[i] / 2
        upper_bound = center + line_widths[i] / 2
        mask = (wavelengths >= lower_bound) & (wavelengths <= upper_bound)
        mean_flux_for_lines.append(mean_flux[mask])
    
    return mean_flux_for_lines

def calc_ccf_template(wavelength, flux1, flux2, velocity):
    ccf = []
    w_low = dopler_shift(wavelength[0], max(velocity))
    w_high = dopler_shift(wavelength[-1], min(velocity))
    delta = wavelength[1] - wavelength[0]
    w_new = np.arange(w_low, w_high, delta/3.)
    for v in velocity:
        w2 = dopler_shift(wavelength, v)
        f1 = np.abs(1 - np.interp(w_new, wavelength, flux1))
        f2 = np.abs(1 - np.interp(w_new, w2, flux2))
        ccf.append(np.sum(f1*f2))
    return ccf

def read_spectrum(infile):
    data = fits.getdata(infile)
    wave, flux = data[0], data[1]
    return wave, flux


def rv_determination(w, f1, f2, velocity, method = 'gaussian', plot = False, ycutoff=-99999):    
    ccf = np.array(calc_ccf_template(w, f1, f2, velocity))
    inds = (ccf >= ycutoff)
    if method == 'gaussian':
        popt, pcov = so.curve_fit(gaussian, velocity[inds], ccf[inds], p0=[velocity[np.argmax(ccf)], np.max(ccf)-np.min(ccf), 20, np.min(ccf)], bounds=([-np.inf, 0, 10, -np.inf], np.inf))
    elif method == 'double_gaussian':
        popt, pcov = so.curve_fit(double_gaussian, velocity[inds], ccf[inds], p0=[velocity[np.argmax(ccf)], np.max(ccf)-np.min(ccf), 20, 1 - velocity[np.argmax(ccf)], (np.max(ccf)-np.min(ccf))/2, 20, np.min(ccf)], bounds=([-np.inf, 0, 10, -np.inf, 0, 10, -np.inf], np.inf))
    elif method == 'quadratic':
        popt, pcov = so.curve_fit(quadratic, velocity[inds], ccf[inds])


    if plot:
        plt.plot(velocity, ccf)
        if method == 'gaussian':
            plt.plot(velocity[inds], gaussian(velocity[inds], *popt))
        elif method == 'double_gaussian':
            plt.plot(velocity[inds], double_gaussian(velocity[inds], *popt))
        elif method == 'quadratic':
            plt.plot(velocity[inds], quadratic(velocity[inds], *popt))
        plt.axvline(popt[0])
        plt.xlabel('Velocity (km/s)')
        plt.show()
    
    rv = popt[0]
    if method == 'gaussian':
        rv_error = np.sqrt(np.diag(pcov))[0]
    elif method == 'double_gaussian':
        rv1 = popt[0]
        rv2 = popt[3]
        rv1_error = np.sqrt(np.diag(pcov))[0]
        rv2_error = np.sqrt(np.diag(pcov))[3]
        rv = [rv1, rv2]
        rv_error = [rv1_error, rv2_error]
    elif method == 'quadratic':
        rv_error = np.sqrt(np.diag(pcov))[0]
    
    return rv, rv_error

def process_time_series(spectra_files, line_list_file, velocity, output_csv='rv_results.csv', log_file='error_log.txt'):
    line_centers, line_widths = read_line_list(line_list_file)
    mean_flux = compute_mean_flux_for_lines(spectra_files, line_centers, line_widths)
    
    if not os.path.exists(output_csv):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Filename', 'UTC', 'MJD-OBS', 'Airmass'] + [f'RV_{center}_Å' for center in line_centers] + [f'RV_error_{center}_Å' for center in line_centers]
            writer.writerow(headers)
    
    with open(log_file, 'w') as log:
        log.write("Error Log\n")

    for infile in tqdm(spectra_files, desc = 'processing:'):

        try:
            wavelength, flux = read_spectrum(infile)
            header = fits.getheader(infile)
            utc_time = header['DATE-OBS']
            mjd_obs = header['MJD-OBS']
            airmass = header['HIERARCH ESO QC AIRM AVG']

            rv_values, rv_errors = [], []
            for i, center in enumerate(line_centers):
                mask = (wavelength > (center - line_widths[i]/2)) & (wavelength < (center + line_widths[i]/2))
                flux_line = flux[mask]
                wavelength_line = wavelength[mask]
                if len(flux_line) == 0:
                    rv_values.append(None)
                    rv_errors.append(None)
                    continue

                ccf = calc_ccf_template(wavelength_line, mean_flux[i], flux_line, velocity)
                try:
                    rv, rv_error = rv_determination(wavelength_line, mean_flux[i], flux_line, velocity)
                    rv_values.append(rv)
                    rv_errors.append(rv_error)
                except RuntimeError:
                    rv_values.append(None)
                    rv_errors.append(None)

            with open(output_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                row = [infile, utc_time, mjd_obs, airmass] + rv_values + rv_errors
                writer.writerow(row)
        except Exception as e:

            with open(log_file, 'a') as log:
                log.write(f"Error with file {infile}: {str(e)}\n")
                print('error logged for:', infile)
            continue

    print(f"RV results saved to {output_csv}")


spectra_files = glob.glob('../manual_normalization/norm/*.fits')
line_list_file = 'linelist.txt'
velocity = np.linspace(-100, 100, 200)
process_timeseries_spectra(spectra_files, line_list_file, velocity)
