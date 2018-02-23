
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.time import Time

from libra import (IRTFTemplate, magnitudes,
                   nirspec_pixel_wavelengths, throughput, trappist1,
                   background, poisson, spitzer_variability,
                   inject_flares, inject_example_flare, transit_duration,
                   Star, trappist1, transit_model, ObservationArchive,
                   trappist1_all_transits, inject_microflares)

sptype_phot = 'M8V'
sptype_spot = 'K0V'
planets = list('bcdefgh')#bh
name = 'TRAPPIST-1'
run_name = 'trappist1_bright2'

import json

observable_transits = json.load(open('libra/data/apt/cycle0/observable_transit_times.json'))

trappist_transits = {k: v for k, v in observable_transits.items() if k.startswith("TRAPPIST")}

wl = nirspec_pixel_wavelengths()
mag = magnitudes['TRAPPIST-1']['J']
exptime = 1*u.s
dataset_kwargs = dict(compression='gzip')

import sys

planet = sys.argv[1]

with ObservationArchive(run_name+'_'+planet, 'w') as obs:
    if 'wavelengths' not in obs.archive:
        obs.archive.create_dataset('wavelengths', data=wl)

    if planet in obs.archive:
        print('deleting old copy')
        del obs.archive[planet]
    group = obs.archive.create_group(planet)

    u1, u2 = trappist1(planet).u
    duration = transit_duration(trappist1(planet))

    spectrum_photo = IRTFTemplate(sptype_phot)
    spectrum_spots = IRTFTemplate(sptype_spot)#spectrum_photo.scale_temperature(delta_teff)

    for midtransit in trappist_transits["{0} {1}".format(name, planet)][:10]:
        print('midtransit', midtransit)
        times = np.arange(midtransit - 1.5*duration,
                          midtransit + 1.5*duration, exptime.to(u.day).value)
        # times = np.arange(midtransit - 1*duration, midtransit, exptime.to(u.day).value)

        #transit = transit_model(times, trappist1(planet))
        transit = trappist1_all_transits(times)

        subgroup = group.create_group("{0}".format(Time(midtransit, format='jd').isot))
        star = Star.with_trappist1_spot_distribution()
        area = star.spotted_area(times)
        fluxes = star.fractional_flux(times)
        flares = inject_flares(wl, times)
        # flares = inject_mi]\m\ares(wl, times)
        spectra = np.zeros((len(times), len(wl)))

        spitzer_var = spitzer_variability(times)

        for i in range(len(times)):
            combined_spectrum = ((1 - area[i]) * spectrum_photo +
                                 area[i] * spectrum_spots)
            spectra[i, :] = poisson(combined_spectrum.n_photons(wl, exptime, mag) *
                                   transit[i] * throughput(wl) * spitzer_var[i] *
                                   (1 + flares[i, :]) + background(wl, exptime))

        # spectral_fluxes = np.sum(spectra, axis=1)
        # plt.scatter(times, spectral_fluxes/spectral_fluxes.mean(),
        #             marker='.', s=4, label='spectrum model')
        # plt.legend()
        # plt.show()
        subgroup.attrs['spot_radii'] = [s.r for s in star.spots]
        subgroup.attrs['spot_contrast'] = star.spots[0].contrast
        subgroup.attrs['t0'] = midtransit

        subgroup.create_dataset('spectra', data=spectra, **dataset_kwargs)
        subgroup.create_dataset('transit', data=transit, **dataset_kwargs)
        subgroup.create_dataset('fluxes', data=fluxes, **dataset_kwargs)
        subgroup.create_dataset('spotted_area', data=area, **dataset_kwargs)
        subgroup.create_dataset('flares', data=1 + flares, **dataset_kwargs)
        subgroup.create_dataset('spitzer_var', data=spitzer_var, **dataset_kwargs)
        subgroup.create_dataset('times', data=times, **dataset_kwargs)
    obs.archive.flush()