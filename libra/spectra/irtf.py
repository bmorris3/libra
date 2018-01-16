"""
This submodule taken from: https://github.com/bmorris3/irtf_templates
"""

import h5py
import os
import astropy.units as u
from astropy.constants import h, c
import numpy as np

from .spectrum import Spectrum1D

__all__ = ['IRTFTemplate']

hdf5_archive_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                                 'irtf_templates.hdf5')

JWST_aperture_area = 25 * u.m**2


class TemplateArchive(object):
    def __init__(self):
        self._hdf5 = None

    @property
    def hdf5(self):
        return h5py.File(hdf5_archive_path, 'r+')

archive = TemplateArchive()


class IRTFTemplate(Spectrum1D):
    def __init__(self, sptype):
        """
        Parameters
        ----------
        sptype : str
            Spectral type of target.
        """
        with archive.hdf5 as f:
            data = f['templates'][sptype][:]
            header = {k: v for k, v in f['templates'][sptype].attrs.items()}

        self.wavelength = data[:, 0] * u.um
        self.flux = data[:, 1] * u.W * u.m**-2 * u.um**-1
        self.error = data[:, 2]
        self.header = header

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

        interped_fluxes = self.interp_flux(wavelengths)

        delta_lambda = np.median(np.diff(wavelengths))
        n_photons_template = (interped_fluxes * wavelengths / h / c *
                              JWST_aperture_area * delta_lambda *
                              exp_time).decompose().value

        relative_target_flux = 10**(0.4 * (self.header['J'] - J))

        return relative_target_flux * n_photons_template
