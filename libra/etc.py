import os
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import astropy.units as u

__all__ = ['throughput', 'quantum_efficiency', 'background', 'poisson']

transmissivity_path = os.path.join(os.path.dirname(__file__), 'data', 'etc',
                                   'NIRSpec_S1600_prism_clear_throughput.csv')

qe_path = os.path.join(os.path.dirname(__file__), 'data', 'etc',
                       'NIRSpec_QE.csv')

background_path = os.path.join(os.path.dirname(__file__), 'data', 'etc',
                               'lineplot_bg_rate.fits')


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
    wl /= 1000  # convert nm to um
    f = interp1d(wl, qe, bounds_error=False, fill_value=1)
    return f(wavelengths)


def background(wavelengths, exp_time):
    """
    Detector background in counts of the NIRSpec S1600 prism/clear mode
    (for BOTS).

    Parameters
    ----------
    wavelength : `~numpy.ndarray`
        Wavelengths in microns

    exp_time : `~astropy.units.Quantity`
        Exposure time

    Returns
    -------
    bg : `~numpy.ndarray`
        Background in e-
    """
    f = fits.getdata(background_path)
    wl, bg = f['WAVELENGTH'], f['bg_rate']
    f = interp1d(wl, bg, bounds_error=False, fill_value=0)
    rate = f(wavelengths) / u.s
    return (rate * exp_time).decompose().value


def poisson(fluxes):
    """
    Add Poisson (counting) noise to ``fluxes``

    Parameters
    ----------
    fluxes : `~numpy.ndarray`
        Fluxes observed

    Returns
    -------
    fluxes : `~numpy.ndarray`
        Fluxes plus poisson uncertainty
    """
    return fluxes + np.sqrt(fluxes) * np.random.randn(len(fluxes))
