import numpy as np

__all__ = ['flare_flux']


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
    flux : `~numpy.ndarray
        Flare fluxes at ``times``
    """
    scaled_times = (times - flare_epoch) / half_width
    flux = np.zeros_like(times)

    rise = (scaled_times < 0) & (scaled_times > -1)
    decay = scaled_times >= 0

    flux[rise] = f_rise(scaled_times[rise])
    flux[decay] = f_decay(scaled_times[decay])
    return flux * delta_f
