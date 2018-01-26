"""
This variability could be granulation, or something else.
"""
import os
import numpy as np
from scipy.interpolate import interp1d

gp_path = os.path.join(os.path.dirname(__file__), 'data',
                       'spitzer_4.5um_gp.txt')

__all__ = ['spitzer_variability']


def spitzer_variability(times, seed=None):
    """
    Mimic unidentified variability observed at 4.5 um in Spitzer.

    Mimic the variability observed at 4.5 um in the Spitzer observations
    of TRAPPIST-1 from Delrez et al. 2018, by interpolating from a gaussian
    process fit to the observations with transits and flares removed..

    Parameters
    ----------
    times : `~numpy.ndarray`
    seed : int or float
        random seed (can be specified for reproducibility)
    Returns
    -------
    f : `~numpy.ndarray`
        Fluxes to be multiplied by your time series
    """
    if seed is not None:
        np.random.seed(seed)

    duration = times[-1] - times[0]
    gp_time, gp_flux = np.loadtxt(gp_path, unpack=True)
    f = interp1d(gp_time, gp_flux, kind='cubic', bounds_error=False)

    if duration > gp_time[-1] - gp_time[0]:
        raise NotImplementedError

    t_start = (gp_time.ptp() - duration) * np.random.rand()
    times_from_zero = times - times.min()

    return f(times_from_zero + t_start) + 1
