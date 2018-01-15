import os
from astropy.io import fits
import astropy.units as u
import numpy as np

from .spectrum import Spectrum1D

__all__ = ['PHOENIXModel']

m2v_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'phoenix',
                        'lte035.0-4.5-0.0a+0.0.BT-Settl.spec.fits')
k2v_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'phoenix',
                        'lte049.0-4.5-0.0a+0.0.BT-Settl.spec.fits')
m8v_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'phoenix',
                        'lte026.0-4.5-0.0a+0.0.BT-Settl.spec.fits')

spt_to_path = dict(m2v=m2v_path, k2v=k2v_path, m8v=m8v_path)


class PHOENIXModel(Spectrum1D):
    def __init__(self, sptype):
        """
        BT-Settle PHOENIX model spectra.

        Parameters
        ----------
        sptype : {"M8V", "M2V", "K2V"}
            Spectral type of host star.
        """

        path = spt_to_path[sptype.lower()]
        f = fits.getdata(path)

        self.wavelength = f['Wavelength'] * u.um
        self.flux = f['Flux']

        # remove nans
        self.wavelength = self.wavelength[~np.isnan(self.flux)]
        self.flux = self.flux[~np.isnan(self.flux)]
