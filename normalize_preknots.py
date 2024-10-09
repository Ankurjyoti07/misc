import numpy as np
import scipy.interpolate as spi
from astropy.io import fits
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_knots(knots_file):
    with open(knots_file, 'r') as f:
        knots = np.array([float(line.strip()) for line in f])
    return knots

def get_mean_flux(wave, flux, knot, width=2.5):
    mask = (wave >= knot - width) & (wave <= knot + width)
    flux_mean = np.median(flux[mask])
    return flux_mean

def normalize_spectrum(wave, flux, knot_points):
    knot_fluxes = np.array([get_mean_flux(wave, flux, knot) for knot in knot_points])
    spline_solution = spi.splrep(knot_points, knot_fluxes, k=2)
    continuum_flux = spi.splev(wave, spline_solution)
    normalized_flux = flux / continuum_flux
    return normalized_flux, continuum_flux

def write_espresso(infile, flux, outfile, wave=None):
    hdul_infile = fits.open(infile)
    hdul_new = fits.HDUList()
    primheader = hdul_infile[0].header.copy()    
    wave = hdul_infile[1].data[0][5]
    hdul_new.append(fits.PrimaryHDU(data=np.vstack((wave, flux)), header=primheader))
    hdul_new.writeto(outfile, overwrite=True)


def plot_flux(wave, flux, continuum_flux, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(wave, flux, label='Unnormalized Flux', color='blue', alpha=0.6)
    plt.plot(wave, continuum_flux, label='Continuum Flux (Spline Fit)', color='red', linestyle='--', linewidth = 0.5)
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

#example
knot_file = '/home/c4011027/PhD_stuff/ESO_proposals/normalized_spectra/knots/ADP.2024-07-05T12:10:57.754_knots.txt'
fits_files = glob.glob('/home/c4011027/PhD_stuff/ESO_proposals/zeta_oph3n/*.fits')
knot_points = read_knots(knot_file)
for fits_file in tqdm(fits_files, desc="Processing FITS files"):
    with fits.open(fits_file) as hdul:
        wave = hdul[1].data[0][5]
        flux = hdul[1].data[0][1]
    normalized_flux, continuum_flux = normalize_spectrum(wave, flux, knot_points)
    output_file = '/home/c4011027/PhD_stuff/ESO_proposals/man_norm/norm/'+f'{fits_file.split("/")[-1].replace(".fits", "")}_norm.fits'
    write_espresso(fits_file, normalized_flux, output_file)
    plot_file = '/home/c4011027/PhD_stuff/ESO_proposals/man_norm/diagnostics/' + f'{fits_file.split("/")[-1].replace(".fits", "")}_normalization.png'
    plot_flux(wave, flux, continuum_flux, plot_file)


