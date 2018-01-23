"""
This submodule taken from: https://github.com/bmorris3/irtf_templates
"""

import os

import astropy.units as u
import h5py
import numpy as np
from astropy.modeling.blackbody import blackbody_lambda

from .spectrum import Spectrum1D
from .spectral_types import spt_to_teff

__all__ = ['IRTFTemplate']

hdf5_archive_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                                 'irtf_templates.hdf5')


class TemplateArchive(object):
    def __init__(self):
        self._hdf5 = None

    @property
    def hdf5(self):
        return h5py.File(hdf5_archive_path, 'r+')

archive = TemplateArchive()


class IRTFTemplate(Spectrum1D):
    def __init__(self, sptype, fill_gaps=True):
        """
        Parameters
        ----------
        sptype : str
            Spectral type of target.
        """
        with archive.hdf5 as f:
            data = f['templates'][sptype.upper()][:]
            header = {k: v for k, v in
                      f['templates'][sptype.upper()].attrs.items()}

        self.wavelength = data[:, 0] * u.um
        self.flux = data[:, 1] * u.W * u.m**-2 * u.um**-1
        self.error = data[:, 2]
        self.header = header

        try:
            self.t_eff = spt_to_teff(sptype)
        except KeyError:
            self.t_eff = None

        if fill_gaps:
            self.fill_gaps()

    def fill_gaps(self):
        """
        Fill gaps in template spectra at strong telluric absorption bands.

        Will do a quadratic fit to the nearby continuum and fill the gap by
        extrapolating the smooth quadratic across the gap.
        """
        normed_template = self.flux# / self.flux.max()
        diffs = np.diff(self.wavelength)
        indices = np.where(diffs > 100*np.median(diffs))[0]
        gap_fillers = []

        for i in indices:
            wl_min, wl_max = self.wavelength[i], self.wavelength[i+1]
            delta_lambda = self.wavelength[i] - self.wavelength[i-1]

            p = np.polyfit(self.wavelength[i-500:i+500],
                           normed_template[i-500:i+500], 2)
            gap_wavelengths = np.arange(wl_min.value+delta_lambda.value,
                                        wl_max.value-delta_lambda.value,
                                        delta_lambda.value) * u.um
            fit = np.polyval(p, gap_wavelengths)
            gap_fillers.append([gap_wavelengths, fit])

        # Extend to 5.3 um
        if self.wavelength.max() < 5.3 * u.um:
            p = np.polyfit(self.wavelength[-1000:], normed_template[-1000:], 1)
            gap_wavelengths = np.arange(self.wavelength.max().value, 5.3,
                                        delta_lambda.value) * u.um
            fit = np.polyval(p, gap_wavelengths)
            gap_fillers.append([gap_wavelengths, fit])

        gap_wls = [i[0] for i in gap_fillers]
        gap_fluxes = [i[1] for i in gap_fillers]
        wl_unit = self.wavelength.unit
        fl_unit = self.flux.unit
        self.wavelength = np.concatenate([self.wavelength] + gap_wls)
        self.flux = np.concatenate([self.flux] + gap_fluxes)

        sort = np.argsort(self.wavelength)
        self.wavelength = u.Quantity(self.wavelength[sort].value, wl_unit)
        self.flux = u.Quantity(self.flux[sort].value, fl_unit)

    def scale_temperature(self, delta_teff, plot=False):
        """
        Scale up or down the flux according to the ratio of blackbodies between
        the effective temperature of the IRTF template spectrum and the
        new effective temperature ``t_eff + delta_teff``.

        Parameters
        ----------
        delta_teff : float
            Change in effective temperature to approximate

        Returns
        -------
        spec : `~libra.IRTFTemplate`
            Scaled spectral template to ``t_eff + delta_teff``
        """

        new_t_eff = self.t_eff + delta_teff
        old_bb = blackbody_lambda(self.wavelength, self.t_eff)
        new_bb = blackbody_lambda(self.wavelength, new_t_eff)

        ratio_bbs = (new_bb / old_bb).value

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(self.wavelength, self.flux, label='init')
            ax.plot(self.wavelength,
                    old_bb * np.max(self.flux) / np.max(old_bb), label='old bb')
            ax.plot(self.wavelength,
                    new_bb * np.max(self.flux) / np.max(old_bb), label='new bb', ls='--')
            ax.plot(self.wavelength, ratio_bbs * self.flux, label='scaled', ls=':')
            ax.legend()

        return Spectrum1D(self.wavelength, ratio_bbs * self.flux,
                          header=self.header, t_eff=new_t_eff)