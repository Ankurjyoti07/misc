import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import scipy.optimize as so

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

def calc_ccf_template(wavelength, flux1, flux2, velocity):
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

def read_spectrum(infile):
    data = fits.getdata(infile)
    wave, flux = data[0], data[1]
    return wave, flux


def rv_determination(w, f1, f2, velocity, fitting_method = 'double_gaussian', plot = True, ycutoff=-99999):    
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

