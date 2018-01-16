import os
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

__all__ = ['throughput', 'quantum_efficiency']

transmissivity_path = os.path.join(os.path.dirname(__file__), 'data', 'etc',
                                   'NIRSpec_S1600_prism_clear_throughput.csv')

qe_path = os.path.join(os.path.dirname(__file__), 'data', 'etc',
                       'NIRSpec_QE.csv')


def throughput(wavelengths):
    """
    Transmission of the NIRSpec S1600 prism/clear mode (for BOTS).

    Parameters
    ----------
    wavelength : `~numpy.ndarray`
        Wavelengths in microns

    Returns
    -------
    trans : `~numpy.ndarray`
        Transmissivity of the NIRSpec S1600 prism/clear mode
    """
    wl, trans = np.loadtxt(transmissivity_path, unpack=True, delimiter=',')
    f = interp1d(wl, trans, bounds_error=False, fill_value=0)
    return f(wavelengths)


def quantum_efficiency(wavelengths):
    """
    QE of the NIRSpec S1600 prism/clear mode (for BOTS).

    Parameters
    ----------
    wavelength : `~numpy.ndarray`
        Wavelengths in microns

    Returns
    -------
    trans : `~numpy.ndarray`
        QE of the NIRSpec S1600 prism/clear mode
    """
    wl, qe = np.loadtxt(qe_path, unpack=True, delimiter=',')
    wl /= 1000 # convert nm to um
    f = interp1d(wl, qe, bounds_error=False, fill_value=1)
    return f(wavelengths)