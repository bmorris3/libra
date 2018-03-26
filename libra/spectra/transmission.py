
import os
import numpy as np
from scipy.interpolate import interp1d


__all__ = ['transmission_spectrum_depths']

transmission_spectrum_path = os.path.join(os.path.dirname(__file__), os.pardir,
                                          'data', 'transmission',
                                          'example_transmission_spectrum.txt')

b_spectrum_path = os.path.join(os.path.dirname(__file__), os.pardir,
                               'data', 'transmission',
                               'smart_spectra_trappist_co2_10bar_b.trnst')
# b_spectrum_path = os.path.join(os.path.dirname(__file__), os.pardir,
#                                'data', 'transmission',
#                                'smart_spectra_trappist_venus_10bar_b.trnst')

path_dict = dict(b=None,
                 c=None,
                 d=None,
                 e=None,
                 f=None,
                 g=None,
                 h=None)

spitzer_wavelengths = np.array([3.6, 4.5])
spitzer_depths = 1e-2 * np.array([0.7070, 0.7277])
spitzer_errors = 1e-2 * np.array([0.0086, 0.0075])


def get_scale_factor(swl, sf):
    model = np.interp(spitzer_wavelengths, swl, sf)
    X = model[:, np.newaxis]
    Sigma = np.diag(spitzer_errors**2)
    inv_sigma = np.linalg.inv(Sigma)
    c = np.linalg.inv(X.T * inv_sigma * X) * X.T * inv_sigma * spitzer_depths
    return c[1, 1]


def transmission_spectrum_depths(planet):
    from .spectrum import nirspec_pixel_wavelengths
    from ..systems import trappist1
    nirspec_wavelengths = nirspec_pixel_wavelengths()

    if path_dict[planet] is not None:
        wavelengths, _, _, depths = np.loadtxt(path_dict[planet],
                                               skiprows=10, unpack=True)

        # Adjust transmission spectrum
        #depths *= 0.9117096418540458

        scale_factor = get_scale_factor(wavelengths, depths)
        depths *= scale_factor

        wavelength_to_depth = interp1d(wavelengths, depths, bounds_error=False,
                                       fill_value=np.mean(depths))
        return wavelength_to_depth(nirspec_wavelengths.value)
    else:
        return trappist1(planet).rp**2 * np.ones_like(nirspec_wavelengths.value)
