
import os
import numpy as np
from scipy.interpolate import interp1d


__all__ = ['transmission_spectrum_depths']

transmission_spectrum_path = os.path.join(os.path.dirname(__file__), os.pardir,
                                          'data', 'transmission',
                                          'example_transmission_spectrum.txt')

b_spectrum_path = os.path.join(os.path.dirname(__file__), os.pardir,
                               'data', 'transmission',
                               't1b_10bar_venus_clear.txt')

path_dict = dict(b=b_spectrum_path,
                 c=None,
                 d=None,
                 e=None,
                 f=None,
                 g=None,
                 h=None)


def transmission_spectrum_depths(planet):
    from .spectrum import nirspec_pixel_wavelengths
    from ..systems import trappist1
    nirspec_wavelengths = nirspec_pixel_wavelengths()

    if path_dict[planet] is not None:
        wavelengths, depths = np.loadtxt(path_dict[planet], unpack=True)

        # Adjust transmission spectrum to include stellar limb-darkening
        u = trappist1('b').u
        ld_factor = 1 - u[0]/3 - u[1]/6
        depths *= ld_factor

        wavelength_to_depth = interp1d(wavelengths, depths, bounds_error=False,
                                       fill_value=np.mean(depths))
        return wavelength_to_depth(nirspec_wavelengths.value)
    else:
        return trappist1(planet).rp**2 * nirspec_wavelengths[:, np.newaxis].value
