import json
import os
import batman
import numpy as np

systems_json_path = os.path.join(os.path.dirname(__file__), 'data',
                                 'systems.json')
magnitudes_path = os.path.join(os.path.dirname(__file__), 'data', 'mags.json')
luminosities_path = os.path.join(os.path.dirname(__file__), 'data',
                                 'luminosities.json')

systems = json.load(open(systems_json_path))

__all__ = ['kepler296', 'kepler62', 'trappist1', 'transit_model', 'magnitudes',
           'luminosities', 'transit_duration']

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