
import os
import numpy as np
from scipy.interpolate import interp1d

__all__ = ['transmission_spectrum']

transmission_spectrum_path = os.path.join(os.path.dirname(__file__), os.pardir,
                                          'data', 'transmission',
                                          'example_transmission_spectrum.txt')

path_dict = dict(b=transmission_spectrum_path,
                 c=transmission_spectrum_path,
                 d=transmission_spectrum_path,
                 e=transmission_spectrum_path,
                 f=transmission_spectrum_path,
                 g=transmission_spectrum_path,
                 h=transmission_spectrum_path)


def transmission_spectrum(planet):
    from .spectrum import nirspec_pixel_wavelengths

    nirspec_wavelengths = nirspec_pixel_wavelengths()

    wavelengths, depths = np.loadtxt(path_dict[planet], unpack=True)
    wavelength_to_depth = interp1d(wavelengths, depths)
    return wavelength_to_depth(nirspec_wavelengths)