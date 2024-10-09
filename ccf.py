import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import scipy.optimize as so
import glob
from astropy.io import fits

def read_espresso(infile):
    print("%s: input file is an espresso spectrum" % infile)
    
    hdul_infile = fits.open(infile)
    wave = hdul_infile[1].data[0][5]
    flux = hdul_infile[1].data[0][1]
    return wave, flux

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

def dopler_shift(w, rv):
    """
    Calculates the Doppler-shifted wavelength given the original wavelength and radial velocity.
    
    Parameters:
    w (array-like): Original wavelength.
    rv (float): Radial velocity.
    
    Returns:
    array-like: Doppler-shifted wavelength.
    """
    w = np.array(w)
    c = 299792.458
    return w * c / (c - rv)


def calc_ccf_linelist(wavelength, flux, velocity, line_list):
    """
    Calculates the cross-correlation function (CCF) between the observed spectrum and a line list.
    
    Parameters:
    wavelength (array-like): Array of wavelengths of the observed spectrum.
    flux (array-like): Array of fluxes of the observed spectrum.
    velocity (array-like): Array of velocities at which to calculate the CCF.
    line_list (array-like): Array of wavelengths of the expected lines.
    
    Returns:
    array-like: Array of CCF values at each velocity.
    """
    ccf = []
    for v in velocity:
        ll = dopler_shift(line_list, v)
        ccf.append(np.sum(np.interp(ll, wavelength, 1 - flux)))
    return ccf


def calc_ccf_template(wavelength, flux1, flux2, velocity):
    """
    Calculates the cross-correlation function (CCF) between the observed spectrum and a template spectrum.
    
    Parameters:
    wavelength (array-like): Array of wavelengths of the observed spectrum.
    flux1 (array-like): Array of fluxes of the observed spectrum.
    flux2 (array-like): Array of fluxes of the template spectrum.
    velocity (array-like): Array of velocities at which to calculate the CCF.
    
    Returns:
    array-like: Array of CCF values at each velocity.
    """

    ccf = []
    w_low = dopler_shift(wavelength[0], max(velocity))
    w_high = dopler_shift(wavelength[-1], min(velocity))
    delta = wavelength[1] - wavelength[0]
    w_new = np.arange(w_low, w_high, delta/10.)
    for v in velocity:
        w2 = dopler_shift(wavelength, v)
        f1 = np.abs(1 - np.interp(w_new, wavelength, flux1))
        f2 = np.abs(1 - np.interp(w_new, w2, flux2))
        ccf.append(np.sum(f1*f2))
    return ccf


def gaussian(x, center, height, std, yoffset):
    """
    Calculates the values of a Gaussian function given an array of x values.
    
    Parameters:
    x (array-like): The values to evaluate the Gaussian.
    center (float): The center of the Gaussian.
    height (float): The height of the Gaussian.
    std (float): The standard deviation of the Gaussian.
    yoffset (float): The y-offset of the Gaussian.
    
    Returns:
    array-like: Array of values of the Gaussian function given the input array.
    """
    return height * np.exp(-1 * (x - center)**2 / (2*std**2)) + yoffset


def double_gaussian(x, center1, height1, std1, center2, height2, std2, yoffset):
    """
    Calculates the values of a double Gaussian function given an array of x values.
    
    Parameters:
    x (array-like): The values to evaluate the double Gaussian.
    center1 (float): The center of the first Gaussian.
    height1 (float): The height of the first Gaussian.
    std1 (float): The standard deviation of the first Gaussian.
    center2 (float): The center of the second Gaussian.
    height2 (float): The height of the second Gaussian.
    std2 (float): The standard deviation of the second Gaussian.
    yoffset (float): The y-offset of the double Gaussian.
    
    Returns:
    array-like: Array of values of the double Gaussian function given the input array.
    """
    return gaussian(x, center1, height1, std1, 0) + gaussian(x, center2, height2, std2, 0) + yoffset


def quadratic(x, q, a, c):
    """
    Calculates the values of a quadratic function given an array of x values.
    
    Parameters:
    x (array-like): The values to evaluate the quadratic function.
    q (float): The quadratic coefficient.
    a (float): The linear coefficient.
    c (float): The constant term.
    
    Returns:
    array-like: Array of values of the quadratic function given the input array.
    """
    b = -2*q*a
    # a = b/(-2*q)
    return a * x**2 + b*x + c


def rv_determination(w, f, line_list=[4471.4802, 4541.59], plot=True, method='gaussian', velocity_range=[-300, 300], ycutoff=-99999):
    """
    Determines the radial velocity of a spectrum using cross-correlation with a line list.
    
    Parameters:
    w (array-like): Array of wavelengths of the observed spectrum.
    f (array-like): Array of fluxes of the observed spectrum.
    line_list (array-like, optional): Array of wavelengths of the expected lines for line list cross-correlation. Default is [4471.4802, 4541.59].
    plot (bool, optional): Whether to plot the cross-correlation function. Default is True.
    method (str, optional): Method for determining the radial velocity. Options are 'gaussian' (default), 'double_gaussian', and 'quadratic'.
    velocity_range (list, optional): Range of velocities at which to calculate the cross-correlation. Default is [-300, 300].
    ycutoff (float, optional): Cutoff value for the cross-correlation function. Default is -99999.
    
    Returns:
    tuple: Tuple containing the radial velocity and its error. The format of the tuple depends on the method used.
    """
    #doppler shift linelist
    line_list = np.array(line_list)
    velocity = np.arange(velocity_range[0], velocity_range[1]+.0001, 0.01)
    inds = (line_list > dopler_shift(w[0], np.max(velocity))) * (line_list < dopler_shift(w[-1], np.min(velocity)))
    ccf = np.array(calc_ccf_linelist(w, f, velocity, line_list[inds]))
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
    #calculate rv
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


line_list, _ = read_line_list('/home/c4011027/PhD_stuff/ESO_proposals/prologs/line_list.txt')
spectrum_file = glob.glob('/home/c4011027/PhD_stuff/ESO_proposals/man_norm/norm/*.fits')
rv_data = np.empty((0, 2))

for spec_file in spectrum_file: 
    w, f = read_espresso(spec_file):
    rv, rv_error = rv_determination(w, f, line_list=line_list, plot=True, method='gaussian', velocity_range=[-300, 300], ycutoff=-0.1)
    rv_data = np.vstack((rv_data, [rv, rv_error]))
np.savetxt('ccf_rvs.csv', rv_data, delimiter=",", header="RV,RV_Error", comments='', fmt='%.6f')
