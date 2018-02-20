import json
import os
import batman
import numpy as np
from copy import deepcopy
systems_json_path = os.path.join(os.path.dirname(__file__), 'data',
                                 'systems.json')
magnitudes_path = os.path.join(os.path.dirname(__file__), 'data', 'mags.json')
luminosities_path = os.path.join(os.path.dirname(__file__), 'data',
                                 'luminosities.json')

systems = json.load(open(systems_json_path))

__all__ = ['kepler296', 'kepler62', 'trappist1', 'transit_model', 'magnitudes',
           'luminosities', 'transit_duration', 'trappist1_all_transits',
           'mask_simultaneous_transits']

magnitudes = json.load(open(magnitudes_path, 'r'))
luminosities = json.load(open(luminosities_path, 'r'))


def batman_generator(star, planet):
    p = batman.TransitParams()
    for attr, value in systems[star][planet].items():
        setattr(p, attr, value)
    return p


def kepler296(planet):
    """
    Get planet properties.

    Parameters
    ----------
    planet : str
        Planet in the system.

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters for planet ``planet``
    """
    return batman_generator('Kepler-296', planet)


def kepler62(planet):
    """
    Get planet properties.

    Parameters
    ----------
    planet : str
        Planet in the system.

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters for planet ``planet``
    """
    return batman_generator('Kepler-62', planet)


def trappist1(planet):
    """
    Get planet properties.

    Parameters
    ----------
    planet : str
        Planet in the system.

    Returns
    -------
    params : `~batman.TransitParams`
        Transit parameters for planet ``planet``
    """
    return batman_generator('TRAPPIST-1', planet)


def transit_model(t, params):
    """
    Generate a transit model at times ``t`` for a planet with propertiesi
    ``params``.

    Parameters
    ----------
    t : `~numpy.ndarray`
        Times in units of days
    params : `~batman.TransitParams()`
        Transiting planet parameter object

    Returns
    -------
    fluxes : `~numpy.ndarray`
        Fluxes at times ``t``
    """
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)
    return flux


def transit_duration(params):
    """
    Roughly approximate the transit duration from other parameters, assuming
    eccentricity = 0.

    Parameters
    ----------
    params : `~batman.TransitParams()`
        Transiting planet parameter object

    Returns
    -------
    duration : float
        Duration in units of days
    """
    b = params.a * np.cos(np.radians(params.inc))
    duration = (params.per / np.pi *
                np.arcsin(np.sqrt((1-params.rp)**2 - b**2) / params.a /
                          np.sin(np.radians(params.inc))))
    return duration


# def trappist1_all_transits(times):
#     all_params = [trappist1(planet) for planet in list('bcdefgh')]
#     flux = np.ones(len(times))
#     for params in all_params:
#         m = batman.TransitModel(params, times)
#         flux += (m.light_curve(params) - 1)
#     return flux
def trappist1_all_transits(times, planet=None):
    from .spectra import nirspec_pixel_wavelengths, transmission_spectrum

    wl = nirspec_pixel_wavelengths()
    all_transit_params = [trappist1(planet) for planet in list('bcdefgh')]
    all_transmission_params = [transmission_spectrum(planet)
                               for planet in list('bcdefgh')]
    flux = np.ones((len(times), len(wl)))

    for params, depths in zip(all_transit_params, all_transmission_params):
        for i, wavelength, depth in zip(range(len(wl)), wl, depths):
            transit_params = deepcopy(params)
            transit_params.rp = depth**0.5
            m = batman.TransitModel(transit_params, times)
            flux[:, i] += (m.light_curve(transit_params) - 1)
    return flux

def mask_simultaneous_transits(times, planet):
    all_params = [trappist1(planet) for planet in list('bcdefgh'.replace(planet, ''))]
    fluxes = []
    for params in all_params:
        m = batman.TransitModel(params, times)
        fluxes.append(m.light_curve(params))

    mask = np.any(np.array(fluxes) != 1, axis=0)

    return np.logical_not(mask)
