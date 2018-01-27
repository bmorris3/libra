import os
import numpy as np
import astropy.units as u
from scipy.stats import poisson
from astropy.modeling.blackbody import blackbody_lambda

__all__ = ['flare_flux', 'inject_flares']

trappist_ffd_path = os.path.join(os.path.dirname(__file__), 'data', 'flares',
                                 'trappist1_ffd_davenport.csv')

trappist_flares_path = os.path.join(os.path.dirname(__file__), 'data', 'flares',
                                    'trappist1_morris.txt')


def f_rise(t_half):
    """
    Davenport+ 2014 Eqn. 1
    """
    return (1 + 1.941 * t_half - 0.175 * t_half**2 -
            2.246 * t_half**3 - 1.125 * t_half**4)


def f_decay(t_half):
    """
    Davenport+ 2014 Eqn. 4
    """
    return 0.6890 * np.exp(-1.600 * t_half) + 0.3030 * np.exp(-0.2783 * t_half)


def flare_flux(times, flare_epoch, delta_f, half_width):
    """
    Generate a flare that follows the flux distribution from
    Davenport et al. 2014.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times of observations
    flare_epoch : float
        Time at maximum flare flux
    delta_f : float
        Maximum flux (measured up from zero)
    half_width : float
        Full-width at half maximum flux, in the same
        time units as ``times``

    Returns
    -------
    flux : `~numpy.ndarray`
        Flare fluxes at ``times``
    """
    scaled_times = (times - flare_epoch) / half_width
    flux = np.zeros_like(times)

    rise = (scaled_times < 0) & (scaled_times > -1)
    decay = scaled_times >= 0

    flux[rise] = f_rise(scaled_times[rise])
    flux[decay] = f_decay(scaled_times[decay])
    return flux * delta_f


def proxima_flare_freq(flare_energy):
    """
    Frequency of flares of given energy on Proxima Cen, from Davenport 2016

    Parameters
    ----------
    flare_energy : float
        Energy of the flare (ergs)

    Returns
    -------
    frequency : float
        Frequency of flares of energy ``flare_energy``
    """
    return 10**(-0.68 * np.log10(flare_energy) + 20.9) * u.d**-1


def proxima_flare_amplitude(flare_energy):
    """
    Relative peak fluxes of flares given energy on Proxima Cen, Davenport 2016

    Parameters
    ----------
    flare_energy : float
        Energy of the flare (ergs)

    Returns
    -------
    amplitude : float
        Peak relative flux during flare
    """
    return 10**(0.48 * np.log10(flare_energy) - 13.6)


def get_flare_durations(observation_duration, verbose=False,
                        min_flare_log_dur=0, max_flare_log_dur=3.5):
                        # min_flare_log_dur=1, max_flare_log_dur=3.5):
    """
    For an observation of length ``observation_duration``,
    return energies of all flares that will occur.

    Assuming we truncate the flare distribution on
    ``(min_flare_log_dur, max_flare_log_dur)``.
    """
    log_dur_range = np.linspace(min_flare_log_dur, max_flare_log_dur, 100)

    best_fit_params = np.array([-0.19222414,  0.49275852])
    log_nu_fit = np.polyval(best_fit_params, log_dur_range)
    log_nu_fit[log_dur_range < 0.5] = 0.45

    counts_decimal = observation_duration * (10**log_nu_fit / u.day)
    counts = np.max([np.floor(counts_decimal).value,
                     np.zeros(len(counts_decimal))], axis=0)

    import matplotlib.pyplot as plt
    plt.plot(log_dur_range, counts_decimal)
    plt.plot(log_dur_range, counts)
    plt.show()

    observed_flare_durs = []

    for integer in np.unique(counts):
        # Minimum dur
        min_dur = np.min(log_dur_range[counts == integer])
        # Maximum dur
        max_dur = np.max(log_dur_range[counts == integer])
        # Randomly draw an integer number of flares from this duration range
        sample_dur = ((max_dur - min_dur)*np.random.rand(int(integer)) + min_dur)
        if verbose:
            print("min={0:.2f}, max={1:.2f}, samples={2}".format(min_dur, max_dur, sample_dur))

        observed_flare_durs.extend(sample_dur)

    return 10**np.array(observed_flare_durs) * u.s


def inject_flares_total_flux(times):
    """
    Inject flares into a transit light curve at ``times``

    Model the flare rate as a Poisson process with rate parameter set by the
    K2 flare frequency - 11 flares in 78.8 days. Draw random variates from that
    Poisson distribution and assign properties to the flares from the K2
    observations.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times of observations

    Returns
    -------
    fluxes : `~numpy.ndarray`
        Fluxes with flares added.
    """
    halfdur, peakflux = np.loadtxt(trappist_flares_path, unpack=True)
    cadence_dt = times[1] - times[0]

    flare_rate = cadence_dt * (11 / 78.8)  # flare rate observed by K2
    r = poisson.rvs(flare_rate, size=times.shape[0])
    n_flares = np.sum(r)
    flare_times = times[r.astype(bool)]

    flux = np.zeros(len(times))

    for i in range(n_flares):
        rand_samp = np.random.randint(0, len(peakflux))
        flux += flare_flux(times, flare_times[i], peakflux[rand_samp] - 1,
                           halfdur[rand_samp])
    return flux


def inject_flares(wavelengths, times):
    """
    Inject flares into a transit light curve at ``times`` at ``wavelengths``

    Model the flare rate as a Poisson process with rate parameter set by the
    K2 flare frequency - 11 flares in 78.8 days. Draw random variates from that
    Poisson distribution and assign properties to the flares from the K2
    observations.

    Assume a 10000 K blackbody for flares.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times of observations

    Returns
    -------
    fluxes : `~numpy.ndarray`
        Fluxes with flares added.
    """
    total_flux = inject_flares_total_flux(times)

    bb = blackbody_lambda(wavelengths, 10000).value
    bb_flux_total = np.sum(bb)  # total flux from flare in bandpass
    return bb / bb_flux_total * total_flux[:, np.newaxis]