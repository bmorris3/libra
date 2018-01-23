import os
import numpy as np
import astropy.units as u

__all__ = ['flare_flux']

trappist_ffd_path = os.path.join(__file__, 'data',
                                 'trappist1_ffd_davenport.csv')


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


def sample_flare_energies(observation_duration, verbose=False,
                          min_flare_energy=28, max_flare_energy=31.5):
    """
    For an observation of length ``observation_duration``,
    return energies of all flares that will occur, assuming we
    truncate the flare distribution on
    ``(min_flare_energy, max_flare_energy)``.
    """
    energy_range = np.linspace(min_flare_energy, max_flare_energy, 100)

    best_fit_params = np.array([-0.19222414,  5.91595501])
    log_nu_fit = np.polyval(best_fit_params, energy_range)

    counts_decimal = observation_duration * (10**log_nu_fit / u.day)
    counts = np.floor(counts_decimal)

    observed_flare_energies = []

    for integer in np.unique(counts):
        # Minimum energy flare
        min_energy = np.min(energy_range[counts == integer])
        # Maximum energy flare
        max_energy = np.max(energy_range[counts == integer])
        # Randomly draw an integer number of flares from this energy range
        sample_energy = ((max_energy - min_energy) *
                         np.random.rand(int(integer)) + min_energy)
        if verbose:
            print("min={0:.2f}, max={1:.2f}, samples={2}"
                  .format(min_energy, max_energy, sample_energy))

        observed_flare_energies.extend(sample_energy)

    return observed_flare_energies


def sample_flares(times):
    observation_duration = times[-1] - times[0]
    energies = sample_flare_energies(observation_duration)


    return fluxes
