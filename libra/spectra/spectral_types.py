import os
from astropy.io import ascii

__all__ = ['spt_to_teff']

mamajek_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                            'spt_mamajek.txt')


def spt_to_teff(spt):
    """
    Convert spectral type (i.e. "M8V") to effective temperature in Kelvin.
    Parameters
    ----------
    spt: str
        Spectral type

    Returns
    -------
    teff : float
        Effective temperature
    """
    mamajek = ascii.read(mamajek_path)
    d = {k: v for k, v in zip(mamajek['SpT'].data, mamajek['Teff'].data)}
    return d[spt.upper()]
