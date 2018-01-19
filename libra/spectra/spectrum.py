from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import astropy.units as u
import h5py
import numpy as np
from astropy.tests.helper import assert_quantity_allclose

__all__ = ['Spectrum1D', 'NIRSpecSpectrum2D', 'ObsArchive']

bg_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'etc',
                       'image_detector.fits')

outputs_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                           'outputs')


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

    @u.quantity_input(new_wavelengths=u.m)
    def interp_flux(self, new_wavelengths):
        f = interp1d(self.wavelength.value, self.flux, kind='linear',
                     bounds_error=False, fill_value=0)
        interped_fluxes = f(new_wavelengths)

        if hasattr(self.flux, 'unit') and self.flux.unit is not None:
            return interped_fluxes * self.flux.unit
        return interped_fluxes

    def __add__(self, other_spectrum):

        if not hasattr(other_spectrum, 'wavelength'):
            raise NotImplementedError()

        interp_flux = (np.interp(other_spectrum.wavelength.value,
                                 self.wavelength.value, self.flux.value) *
                       self.flux.unit)

        return Spectrum1D(other_spectrum.wavelength,
                          interp_flux + other_spectrum.flux,
                          header=[self.header, other_spectrum.header])

    def __rmul__(self, multiplier):
        if not np.isscalar(multiplier):
            raise NotImplementedError()

        return Spectrum1D(self.wavelength, multiplier * self.flux,
                          header=self.header)


class NIRSpecSpectrum2D(object):
    def __init__(self):
        self.image = fits.getdata(bg_path)


class ObsArchive(object):
    def __init__(self, fname):
        self.path = os.path.join(outputs_dir, fname)
        self.archive = None

    def __enter__(self):
        self.archive = h5py.File(self.path, 'rw+')
        return self

    def __exit__(self, *args):
        self.archive.close()
        pass
