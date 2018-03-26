from scipy.interpolate import interp1d
import os
from astropy.io import fits
import astropy.units as u
import h5py
import numpy as np
from astropy.constants import h, c
import matplotlib.pyplot as plt

__all__ = ['Spectrum1D', 'ObservationArchive', 'nirspec_pixel_wavelengths',
           'Simulation', 'Spectra1D', 'n_photons']

bg_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'etc',
                       'image_detector.fits')

wl_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'etc',
                       'lineplot_wave_pix.fits')

outputs_dir_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                                'outputs')

JWST_aperture_area = 25 * u.m**2


def nirspec_pixel_wavelengths():
    return fits.getdata(wl_path)['WAVELENGTH'] * u.um


class Spectrum1D(object):
    def __init__(self, wavelength, flux, error=None, header=None, t_eff=None):
        self.wavelength = wavelength
        self.flux = flux
        self.error = error
        self.header = header
        self.t_eff = t_eff

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        ax.plot(self.wavelength, self.flux, **kwargs)

        return ax

    @u.quantity_input(new_wavelengths=u.m)
    def interp_flux(self, new_wavelengths):
        f = interp1d(self.wavelength.value, self.flux, kind='linear',
                     bounds_error=False, fill_value=0)
        interped_fluxes = f(new_wavelengths)

        if hasattr(self.flux, 'unit') and self.flux.unit is not None:
            return interped_fluxes * self.flux.unit
        return interped_fluxes

    def __add__(self, other_spectrum):

        # if not hasattr(other_spectrum, 'wavelength'):
        #     raise NotImplementedError()
        #
        # interp_flux = (np.interp(other_spectrum.wavelength.value,
        #                          self.wavelength.value, self.flux.value) *
        #                self.flux.unit)

        if not other_spectrum.wavelength == self.wavelength:
            raise NotImplementedError()

        return Spectrum1D(other_spectrum.wavelength,
                          self.flux + other_spectrum.flux,
                          header=self.header)

    def __rmul__(self, multiplier):
        # if not np.isscalar(multiplier):
        #     raise NotImplementedError()

        return Spectrum1D(self.wavelength, multiplier * self.flux,
                          header=self.header)


def n_photons(wavelengths, fluxes, J, header, n_groups):
    """
    Estimate the number of photons received from a target with J magnitude
    ``J`` over exposure time ``exp_time``.

    Parameters
    ----------
    wavelengths : `~astropy.units.Quantity`
        Wavelengths to test
    exp_time : `~astropy.units.Quantity`
        Exposure time
    J : float
        J-band magnitude of the target
    n_groups : int
        Number of groups per integration
    Returns
    -------
    fluxes : `~numpy.ndarray`
        Counts that reach the telescope at each wavelength

    References
    ----------
    [1] https://jwst-docs.stsci.edu/display/JTI/NIRSpec+Bright+Object+Time-Series+Spectroscopy
    """
    if not hasattr(fluxes, 'unit'):
        raise NotImplementedError("Flux must have units")

    if 2 > n_groups:
        raise ValueError("Group # must be >= 2")

#    efficiency = (n_groups - 1) / (n_groups + 1)

    exp_time = (n_groups - 1) * 0.22616 * u.s

    delta_lambda = np.nanmedian(np.diff(wavelengths))
    n_photons_template = (fluxes * wavelengths / h / c *
                          JWST_aperture_area * delta_lambda *
                          exp_time).decompose().value

    relative_target_flux = 10**(0.4 * (float(header['J']) - J))

    return relative_target_flux * n_photons_template



class Spectra1D(object):
    def __init__(self, wavelength, flux, error=None, header=None, t_eff=None):
        self.wavelength = wavelength
        self.flux = flux
        self.error = error
        self.header = header
        self.t_eff = t_eff

    @u.quantity_input(new_wavelengths=u.m)
    def interp_flux(self, new_wavelengths):
        f = interp1d(self.wavelength.value, self.flux, kind='linear',
                     bounds_error=False, fill_value=0, axis=1)
        interped_fluxes = f(new_wavelengths)

        if hasattr(self.flux, 'unit') and self.flux.unit is not None:
            return interped_fluxes * self.flux.unit
        return interped_fluxes

    def n_photons(self, wavelengths, exp_time, J):
        """
        Estimate the number of photons received from a target with J magnitude
        ``J`` over exposure time ``exp_time``.

        Parameters
        ----------
        wavelengths : `~astropy.units.Quantity`
            Wavelengths to test
        exp_time : `~astropy.units.Quantity`
            Exposure time
        J : float
            J-band magnitude of the target

        Returns
        -------
        fluxes : `~numpy.ndarray`
            Counts that reach the telescope at each wavelength
        """
        if not hasattr(self.flux, 'unit'):
            raise NotImplementedError("Flux must have units")

        interped_fluxes = self.interp_flux(wavelengths)

        delta_lambda = np.nanmedian(np.diff(wavelengths))
        n_photons_template = (interped_fluxes * wavelengths / h / c *
                              JWST_aperture_area * delta_lambda *
                              exp_time).decompose().value

        relative_target_flux = 10**(0.4 * (float(self.header['J']) - J))

        return relative_target_flux * n_photons_template


class ObservationArchive(object):
    def __init__(self, fname, mode='r', outputs_dir=None):

        if outputs_dir is None:
            outputs_dir = outputs_dir_path

        self.path = os.path.join(outputs_dir, fname + '.hdf5')
        self.target_name = fname
        self.archive = None
        self.mode = mode

    def __enter__(self):
        self.archive = h5py.File(self.path, self.mode)

        planets = [i for i in list('bcdefgh') if i in self.archive]

        for planet in planets:
            simulations = []
            for iteration in self.archive[planet]:
                attrs = dict(self.archive[planet][iteration].attrs)
                simulations.append(Simulation(self.archive[planet][iteration],
                                              attrs=attrs,
                                              path="/{0}/{1}/".format(planet,
                                                                      iteration)))
            setattr(self, planet, simulations)

        return self

    def __exit__(self, *args):
        self.archive.close()


class Simulation(object):
    def __init__(self, observation, attrs=None, path=None):
        self.observation = observation
        self.attrs = attrs
        self.path = path

    @property
    def times(self):
        return self.observation['times'][:]

    @property
    def areas(self):
        return self.observation['spotted_area'][:]

    @property
    def spitzer_var(self):
        return self.observation['spitzer_var'][:]

    @property
    def granulation(self):
        return self.observation['granulation'][:]

    @property
    def flares(self):
        return self.observation['flares'][:]

    @property
    def fluxes(self):
        return self.observation['fluxes'][:]

    @property
    def spectra(self):
        return self.observation['spectra'][:]

    @property
    def samples_depth(self):
        return self.observation['samples/depth'][:]

    @property
    def samples_t0(self):
        return self.observation['samples/t0'][:]

    @property
    def samples_amp(self):
        return self.observation['samples/amp'][:]

    @property
    def samples_log_S0(self):
        return self.observation['samples/log_S0'][:]

    @property
    def samples_log_omega0(self):
        return self.observation['samples/log_omega0'][:]

    @property
    def samples_duration(self):
        return self.observation['samples/duration'][:]

    @property
    def samples_b(self):
        return self.observation['samples/b'][:]

    # @property
    # def samples_log_a(self):
    #     return self.observation['samples/log_a'][:]

    @property
    def samples_median(self):
        samples = (self.samples_log_S0, self.samples_log_omega0,
                   self.samples_amp, self.samples_depth,
                   self.samples_t0, self.samples_duration, self.samples_b)
        return np.array([np.median(s) for s in samples])

    def plot(self):
        wl = nirspec_pixel_wavelengths()

        fig, ax = plt.subplots(2, 5, figsize=(14, 6))
        ax[0, 0].plot(self.times, self.areas)
        ax[0, 0].set(xlabel='Time', ylabel='Spotted area')

        ax[0, 1].plot(self.times, self.fluxes)
        ax[0, 1].set(xlabel='Time', ylabel='Stellar flux')

        monochromatic_flares = np.sum(self.flares, axis=1)
        monochromatic_flares /= np.median(monochromatic_flares)
        ax[0, 2].plot(self.times, monochromatic_flares)
        ax[0, 2].set(xlabel='Time', ylabel='Flare flux')

        ax[0, 3].plot(self.times, self.spitzer_var)
        ax[0, 3].set(xlabel='Time', ylabel='Spitzer var.')

        ax[0, 4].plot(self.times, self.transit)
        ax[0, 4].set(xlabel='Time', ylabel='Transit')

        ax[1, 0].imshow(self.spectra, extent=[0.6, 5.3, 0, self.times.ptp()])
        ax[1, 0].set(title='Spectrophotometry', aspect=3/(self.times.ptp()),
                     xlabel='Wavelength [$\mu$m]', ylabel='Time [d]')

        short_bin = np.sum(self.spectra[:, :100], axis=1)
        mid_bin = np.sum(self.spectra[:, 100:200], axis=1)
        long_bin = np.sum(self.spectra[:, 200:], axis=1)
        ax[1, 1].plot(self.times, long_bin/long_bin.max(), ',', color='C0',
                      label=r'{0:.2f}-{1:.2f} $\mu$m'
                      .format(wl[0].value, wl[100].value))
        ax[1, 2].plot(self.times, mid_bin/mid_bin.max(), ',', color='C2',
                      label=r'{0:.2f}-{1:.2f} $\mu$m'
                      .format(wl[100].value, wl[200].value))
        ax[1, 3].plot(self.times, short_bin/short_bin.max(), ',', color='r',
                      label=r'{0:.2f}-{1:.2f} $\mu$m'
                      .format(wl[200].value, wl[-1].value))

        ax[1, 4].plot(self.times, np.sum(self.spectra, axis=1), ',')
        ax[1, 4].set(xlabel='Time', ylabel='NIRSpec counts',
                     title='Band-integrated')

        for axis in [ax[1, 1], ax[1, 2], ax[1, 3]]:
            axis.get_shared_y_axes().join(axis, ax[1, 1])
            axis.legend()
            axis.set_xlabel('Time')
            axis.set_ylabel('Flux')

        fig.tight_layout()

        return fig, ax