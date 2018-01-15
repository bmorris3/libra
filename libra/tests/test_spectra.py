import astropy.units as u
import numpy as np
from ..spectra import IRTFTemplate, PHOENIXModel


def test_irtf():
    spectra = [IRTFTemplate('M8V'), IRTFTemplate('M2V'), IRTFTemplate('K2V')]

    for spectrum in spectra:
        assert hasattr(spectrum.wavelength, 'unit')
        assert not hasattr(spectrum.flux, 'unit')

        blue_flux = spectrum.flux[spectrum.wavelength < 2.5*u.um]
        red_flux = spectrum.flux[spectrum.wavelength > 2.5*u.um]

        assert np.mean(blue_flux) > np.mean(red_flux)