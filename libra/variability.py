"""
This variability could be granulation, or something else.
"""
import os
import numpy as np
from scipy.interpolate import interp1d
from celerite import terms
import celerite

gp_path_T1 = os.path.join(os.path.dirname(__file__), 'data',
                          'spitzer_4.5um_gp.txt')

gp_path_62 = os.path.join(os.path.dirname(__file__), 'data',
                          'K62_variability.txt')

__all__ = ['spitzer_variability', 'k62_variability', 'k296_variability',
           'trappist1_variability']


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

    duration = times.max() - times.min()
    gp_time, gp_flux = np.loadtxt(gp_path_T1, unpack=True)
    f = interp1d(gp_time, gp_flux, kind='linear', bounds_error=False,
                 fill_value=0)

    if duration > gp_time.max() - gp_time.min():
        raise NotImplementedError()

    t_start = (gp_time.ptp() - duration) * np.random.rand()
    times_from_zero = times - times.min()

    return f(times_from_zero + t_start) + 1


def k62_variability(times, seed=None):
    # alpha = 0.800231836683
    # log_a = np.log(np.exp(-15.45759496) * alpha)
    # log_c = -1.2202536360766596
    # # log_sigma = -8.8461704
    #
    # # kernel = (terms.JitterTerm(log_sigma=log_sigma) +
    # #           terms.RealTerm(log_a=log_a, log_c=log_c))
    #
    # kernel = terms.RealTerm(log_a=log_a, log_c=log_c)
    #
    # gp = celerite.GP(kernel, mean=0, fit_white_noise=True, fit_mean=True)
    # gp.compute(times)
    #
    # sample = gp.sample()
    # sample -= np.median(sample)
    # return sample + 1

    if seed is not None:
        np.random.seed(seed)

    duration = times.max() - times.min()
    gp_time, gp_flux = np.loadtxt(gp_path_62, unpack=True)
    f = interp1d(gp_time, gp_flux, kind='linear', bounds_error=False,
                 fill_value=0)

    if duration > gp_time.max() - gp_time.min():
        raise NotImplementedError()

    t_start = (gp_time.ptp() - duration) * np.random.rand()
    times_from_zero = times - times.min()

    return f(times_from_zero + t_start)


def k296_variability(times):
    alpha = 0.854646217641
    log_a = np.log(np.exp(-13.821195) * alpha)
    log_c = -1.0890621571818671
    # log_sigma = -7.3950524

    # kernel = (terms.JitterTerm(log_sigma=log_sigma) +
    #           terms.RealTerm(log_a=log_a, log_c=log_c))

    kernel = terms.RealTerm(log_a=log_a, log_c=log_c)

    gp = celerite.GP(kernel, mean=0, fit_white_noise=True, fit_mean=True)
    gp.compute(times)

    sample = gp.sample()
    sample -= np.median(sample)
    return sample + 1


def trappist1_variability(times):
    alpha = 0.973460343001
    log_a = np.log(np.exp(-26.88111923) * alpha)
    log_c = -1.0890621571818671
    # log_sigma = -5.6551601053314622

    # kernel = (terms.JitterTerm(log_sigma=log_sigma) +
    #           terms.RealTerm(log_a=log_a, log_c=log_c))

    kernel = terms.RealTerm(log_a=log_a, log_c=log_c)

    gp = celerite.GP(kernel, mean=0, fit_white_noise=True, fit_mean=True)
    gp.compute(times)

    sample = gp.sample()
    sample -= np.median(sample)
    return sample + 1
