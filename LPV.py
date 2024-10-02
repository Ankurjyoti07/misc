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
                width = 14.0
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
    sorted_obs_dates = [s[1] for s in spectra_with_dates]
    return sorted_spectra_files, sorted_obs_dates

def animate_spectra(spectra_files, spectra_dates, line_centers, line_widths, save_movie=False):
    """
    spectra_files: list of spectra files
    line_centers: wavelength array to plot
    line_widths: line widths specified in the list
    save_movie: save the animation as a movie file
    """
    num_lines = len(line_centers)
    
    grid_size = math.ceil(math.sqrt(num_lines))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 8))
    axes = axes.flatten()  # flatten in case grid_size > 1

    def init():
        for ax in axes:
            ax.clear()
            ax.set_xlim(min(line_centers) - max(line_widths), max(line_centers) + max(line_widths))
            ax.set_ylim(0.65, 1.2)  #adjust depending on flux scale
            ax.set_xlabel('Wavelength [Å]')
            ax.set_ylabel('Flux')

    #animate movie
    def update(frame):
        for i, (center, width) in enumerate(zip(line_centers, line_widths)):
            ax = axes[i]
            ax.clear()
            wavelengths, flux = read_spectrum(spectra_files[frame])
            line_mask = (wavelengths >= center - width) & (wavelengths <= center + width)
            ax.plot(wavelengths[line_mask], flux[line_mask])
            ax.set_xlim(center - width, center + width)
            ax.set_ylim(0.65, 1.2)  #adjust depending on flux scale
            ax.set_title(f'Line Center: {center:.1f} Å')

        fig.suptitle(f'Spectral Evolution - Frame {frame+1}/{len(spectra_files)} - MJD {spectra_dates[frame]}')
        plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=len(spectra_files), init_func=init, blit=False, repeat=False)

    if save_movie:
        ani.save('spectra_evolution_2.mp4', writer='ffmpeg', fps=3)  #save movie
    else:
        plt.show()


def main():
    spectra_dir = input("Enter the directory containing the spectra files: ")
    spectra_files = sorted(glob.glob(os.path.join(spectra_dir, '*.fits')))  #adjust .fits/ format
    
    #linelist_file = input("Enter the path to the linelist file: ")
    line_centers, line_widths = read_line_list('/home/c4011027/PhD_stuff/ESO_proposals/prologs/line_list.txt')

    save_movie = input("Do you want to save the movie? (yes/no): ").lower() == 'yes'
    animate_spectra(sort_spectra(spectra_files)[0][1240:1300], sort_spectra(spectra_files)[1][1240:1300], line_centers, line_widths, save_movie=True)

if __name__ == "__main__":
    main()