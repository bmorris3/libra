
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.time import Time

from libra import (IRTFTemplate, magnitudes,
                   nirspec_pixel_wavelengths, throughput,
                   background, poisson, spitzer_variability,
                   inject_flares, inject_example_flare, transit_duration,
                   Star, kepler296, transit_model, ObservationArchive,
                   k296_all_transits, Spectra1D, trappist_out_of_transit,
                   n_photons)

sptype_phot = 'K7V'
sptype_spot = 'K0V'
planets = list('b')
name = 'Kepler-296'
planet = 'b'

import sys
i = sys.argv[1]
run_name = 'tyler{0}'.format(i)
#transit_params = kepler296

duration = transit_duration(kepler296(planet))

midtransit = mt = Time('1991-02-21 8:00').jd + 100*np.random.rand()
print(100*np.random.rand(), midtransit)
#transits = {"Kepler-296 b": [mt]}
params = kepler296('b')

rotation_period = 2 * np.random.rand() + 1
params.rp = params.rp * (0.3 * np.random.rand() + 0.4)
params.t0 = mt

np.savetxt(run_name+'.txt', [params.rp, params.t0, rotation_period])

wl = nirspec_pixel_wavelengths()
mag = magnitudes['Kepler-296']['J']
exptime = 5*u.s
dataset_kwargs = dict(compression='gzip')


with ObservationArchive(run_name+'_' + planet, 'w') as obs:
    if 'wavelengths' not in obs.archive:
        obs.archive.create_dataset('wavelengths', data=wl)

    if planet in obs.archive:
        print('deleting old copy')
        del obs.archive[planet]
    group = obs.archive.create_group(planet)

    u1, u2 = params.u
    #duration = transit_duration(transit_params(planet))

    spectrum_photo = IRTFTemplate(sptype_phot)
    spectrum_spots = IRTFTemplate(sptype_spot)

    print('midtransit', midtransit)
    times = np.arange(midtransit - (8*np.random.rand() + 2)*duration,
                      midtransit + (8*np.random.rand() + 2)*duration, exptime.to(u.day).value)
    # times = np.arange(midtransit - 1*duration, midtransit, exptime.to(u.day).value)

    transit = transit_model(times, params)#all_transits(times)

    subgroup = group.create_group("{0}".format(Time(midtransit, format='jd').isot))
    star = Star.with_k296_spot_distribution()
    star.rotation_period = rotation_period * u.day
    area = star.spotted_area(times)
    fluxes = star.fractional_flux(times)

    spitzer_var = spitzer_variability(times)[:, np.newaxis]

    spectrum_photo_flux = spectrum_photo.interp_flux(wl)
    spectrum_spots_flux = spectrum_spots.interp_flux(wl)

    combined_spectra = ((transit[:, np.newaxis] - area[:, np.newaxis]) *
                         spectrum_photo_flux + area[:, np.newaxis] *
                         spectrum_spots_flux)

    spectra = poisson(n_photons(wl, combined_spectra, exptime, mag,
                                spectrum_photo.header) * spitzer_var *
                      throughput(wl)[np.newaxis, :]
                      + background(wl, exptime)[np.newaxis, :])

    # spectral_fluxes = np.sum(spectra, axis=1)
    # plt.scatter(times, spectral_fluxes/spectral_fluxes.mean(),
    #             marker='.', s=4, label='spectrum model')
    # plt.legend()
    # plt.show()
    # subgroup.attrs['spot_radii'] = [s.r for s in star.spots]
    # subgroup.attrs['spot_contrast'] = star.spots[0].contrast
    # subgroup.attrs['t0'] = midtransit

    subgroup.create_dataset('spectra', data=spectra, **dataset_kwargs)
    # subgroup.create_dataset('transit', data=transit, **dataset_kwargs)
    # subgroup.create_dataset('fluxes', data=fluxes, **dataset_kwargs)
    # subgroup.create_dataset('spotted_area', data=area, **dataset_kwargs)
    #subgroup.create_dataset('flares', data=1 + flares, **dataset_kwargs)
    # subgroup.create_dataset('spitzer_var', data=spitzer_var, **dataset_kwargs)
    subgroup.create_dataset('times', data=times, **dataset_kwargs)
    obs.archive.flush()