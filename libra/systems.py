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
           'mask_simultaneous_transits_trappist', 'trappist_out_of_transit',
           'k296_all_transits', 'mask_simultaneous_transits_k296',
           'mask_simultaneous_transits_k296', 'k62_all_transits', 'kepler1600']

magnitudes = json.load(open(magnitudes_path, 'r'))
luminosities = json.load(open(luminosities_path, 'r'))
supersample_factor = 3


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


def kepler1600(planet):
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
    return batman_generator('Kepler-1600', planet)


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
    m = batman.TransitModel(params, t, supersample_factor=supersample_factor,
                            exp_time=t[1]-t[0])
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


def trappist1_all_transits(times):
    from .spectra import nirspec_pixel_wavelengths, transmission_spectrum_depths

    wl = nirspec_pixel_wavelengths()
    all_transit_params = [trappist1(planet) for planet in list('bcdefgh')]
    all_transmission_depths = [transmission_spectrum_depths(planet)
                               for planet in list('bcdefgh')]
    flux = np.ones((len(times), len(wl)))

    for params, depths in zip(all_transit_params, all_transmission_depths):
        for i, wavelength, depth in zip(range(len(wl)), wl, depths):
            transit_params = deepcopy(params)
            transit_params.rp = depth**0.5
            m = batman.TransitModel(transit_params, times,
                                    supersample_factor=supersample_factor,
                                    exp_time=times[1]-times[0])
            flux[:, i] += (m.light_curve(transit_params) - 1)
    return flux


def k296_all_transits(times):
    from .spectra import nirspec_pixel_wavelengths

    wl = nirspec_pixel_wavelengths()
    all_transit_params = [kepler296(planet) for planet in list('bcdef')]
    flux = np.ones((len(times), len(wl)))

    for i, params in enumerate(all_transit_params):
        m = batman.TransitModel(params, times,
                                supersample_factor=supersample_factor,
                                exp_time=times[1]-times[0])
        flux += (m.light_curve(params) - 1)[:, np.newaxis]
    return flux


def k62_all_transits(times):
    from .spectra import nirspec_pixel_wavelengths

    wl = nirspec_pixel_wavelengths()
    all_transit_params = [kepler62(planet) for planet in list('bcdef')]
    flux = np.ones((len(times), len(wl)))

    for i, params in enumerate(all_transit_params):
        m = batman.TransitModel(params, times,
                                supersample_factor=supersample_factor,
                                exp_time=times[1]-times[0])
        flux += (m.light_curve(params) - 1)[:, np.newaxis]
    return flux


def mask_simultaneous_transits_trappist(times, planet):
    all_params = [trappist1(planet) for planet in list('bcdefgh'.replace(planet, ''))]
    fluxes = []
    for params in all_params:
        m = batman.TransitModel(params, times)
        fluxes.append(m.light_curve(params))

    mask = np.any(np.array(fluxes) != 1, axis=0)

    return np.logical_not(mask)


def mask_simultaneous_transits_k296(times, planet):
    all_params = [kepler296(planet) for planet in list('bcdef'.replace(planet, ''))]
    fluxes = []
    for params in all_params:
        m = batman.TransitModel(params, times)
        fluxes.append(m.light_curve(params))

    mask = np.any(np.array(fluxes) != 1, axis=0)

    return np.logical_not(mask)


def mask_simultaneous_transits_k62(times, planet):
    all_params = [kepler62(planet) for planet in list('bcdef'.replace(planet, ''))]
    fluxes = []
    for params in all_params:
        m = batman.TransitModel(params, times)
        fluxes.append(m.light_curve(params))

    mask = np.any(np.array(fluxes) != 1, axis=0)

    return np.logical_not(mask)


def trappist_out_of_transit(times):
    all_params = [trappist1(planet) for planet in list('bcdefgh')]
    fluxes = []
    for params in all_params:
        m = batman.TransitModel(params, times)
        fluxes.append(m.light_curve(params))
    mask = np.all(np.array(fluxes) == 1, axis=0)
    return mask
