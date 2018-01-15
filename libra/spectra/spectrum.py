import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

__all__ = ['Spectrum1D']


class Spectrum1D(object):
    def __init__(self, wavelength, flux, error=None, header=None):
        self.wavelength = wavelength
        self.flux = flux
        self.error = error
        self.header = header

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.plot(self.wavelength, self.flux)

        return ax

    def interp_flux(self, new_wavelengths):
        f = interp1d(self.wavelength.value, self.flux, kind='linear',
                     bounds_error=False, fill_value=0)
        return f(new_wavelengths)
