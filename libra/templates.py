"""
This submodule taken from: https://github.com/bmorris3/irtf_templates
"""

import matplotlib.pyplot as plt
import h5py
import os

__all__ = ['IRTFTemplate']

hdf5_archive_path = os.path.join(os.path.dirname(__file__), 'data',
                                 'irtf_templates.hdf5')


class TemplateArchive(object):
    def __init__(self):
        self._hdf5 = None

    @property
    def hdf5(self):
        return h5py.File(hdf5_archive_path, 'r+')

archive = TemplateArchive()


class IRTFTemplate(object):
    def __init__(self, sptype):
        """
        Parameters
        ----------
        sptype : str
            Spectral type of target
        """
        with archive.hdf5 as f:
            data = f['templates'][sptype][:]
            header = {k: v for k, v in f['templates'][sptype].attrs.items()}

        self.wavelength = data[:, 0]
        self.flux = data[:, 1]
        self.error = data[:, 2]
        self.header = header

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.plot(self.wavelength, self.flux)

        return ax

