from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import astropy.units as u
import h5py
import numpy as np
from astropy.constants import h, c

__all__ = ['Spectrum1D', 'NIRSpecSpectrum2D', 'ObservationArchive',
           'nirspec_pixel_wavelengths']

bg_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'etc',
                       'image_detector.fits')

wl_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'etc',
                       'lineplot_wave_pix.fits')

outputs_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                           'outputs')

JWST_aperture_area = 25 * u.m**2


def nirspec_pixel_wavelengths():
    return fits.getdata(wl_path)['WAVELENGTH'] * u.um


class Spectrum1D(object):
    def __init__(self, wavelength, flux, error=None, header=None, t_eff=None):
        self.wavelength = wavelength
        self.flux = flux
        self.error = error
        self.header = header
        self.t_eff = t_eff

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        ax.plot(self.wavelength, self.flux, **kwargs)

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
                          header=self.header)

    def __rmul__(self, multiplier):
        if not np.isscalar(multiplier):
            raise NotImplementedError()

        return Spectrum1D(self.wavelength, multiplier * self.flux,
                          header=self.header)

    def n_photons(self, wavelengths, exp_time, J):
        """
        Estimate the number of photons received from a target with J magnitude
        ``J`` over exposure time ``exp_time``.

        Parameters
        ----------
        wavelengths : `~astropy.units.Quantity`
            Wavelengths to test
        exp_time : `~astropy.units.Quantity`
            Exposure time
        J : float
            J-band magnitude of the target

        Returns
        -------
        fluxes : `~numpy.ndarray`
            Counts that reach the telescope at each wavelength
        """
        if not hasattr(self.flux, 'unit'):
            raise NotImplementedError("Flux must have units")

        interped_fluxes = self.interp_flux(wavelengths)

        delta_lambda = np.nanmedian(np.diff(wavelengths))
        n_photons_template = (interped_fluxes * wavelengths / h / c *
                              JWST_aperture_area * delta_lambda *
                              exp_time).decompose().value

        relative_target_flux = 10**(0.4 * (float(self.header['J']) - J))

        return relative_target_flux * n_photons_template


class NIRSpecSpectrum2D(object):
    def __init__(self):
        self.image = fits.getdata(bg_path)


class ObservationArchive(object):
    def __init__(self, fname, mode='r'):
        self.path = os.path.join(outputs_dir, fname + '.hdf5')
        self.target_name = fname
        self.archive = None
        self.mode = mode

    def __enter__(self):
        self.archive = h5py.File(self.path, self.mode)
        return self

    def __exit__(self, *args):
        self.archive.close()

    def times(self, planet, iteration):
        return self.archive[planet][iteration]['times']

    def areas(self, planet, iteration):
        return self.archive[planet][iteration]['spotted_area']

    def spitzer_var(self, planet, iteration):
        return self.archive[planet][iteration]['spitzer_var']

    def flares(self, planet, iteration):
        return self.archive[planet][iteration]['flares']

    def fluxes(self, planet, iteration):
        return self.archive[planet][iteration]['fluxes']