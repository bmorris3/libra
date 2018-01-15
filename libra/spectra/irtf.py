"""
This submodule taken from: https://github.com/bmorris3/irtf_templates
"""

import h5py
import os
import astropy.units as u
from .spectrum import Spectrum1D

__all__ = ['IRTFTemplate']

hdf5_archive_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                                 'irtf_templates.hdf5')


class TemplateArchive(object):
    def __init__(self):
        self._hdf5 = None

    @property
    def hdf5(self):
        return h5py.File(hdf5_archive_path, 'r+')

archive = TemplateArchive()


class IRTFTemplate(Spectrum1D):
    def __init__(self, sptype):
        """
        Parameters
        ----------
        sptype : str
            Spectral type of target.
        """
        with archive.hdf5 as f:
            data = f['templates'][sptype][:]
            header = {k: v for k, v in f['templates'][sptype].attrs.items()}

        self.wavelength = data[:, 0] * u.um
        self.flux = data[:, 1]  # * u.W * u.m**-2 * u.um**-1
        self.error = data[:, 2]
        self.header = header
