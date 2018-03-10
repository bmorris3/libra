
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

sptype_phot = 'M1V'
sptype_spot = 'M3V'
secondary_sptype = 'M3V'
dilution_factor = 0.785
planets = list('bcdef')
name = 'Kepler-296'
run_name = 'k296'
transit_params = kepler296
all_transits = k296_all_transits
import json

observable_transits = json.load(open('libra/data/apt/cycle1/observable_transit_times.json'))

transits = {k: v for k, v in observable_transits.items() if k.startswith("Kepler-296")}

wl = nirspec_pixel_wavelengths()
mag = magnitudes['Kepler-296']['J']
exptime = 5*u.s
dataset_kwargs = dict(compression='gzip')

import sys

planet = sys.argv[1]

with ObservationArchive(run_name+'_' + planet, 'w') as obs:
    if 'wavelengths' not in obs.archive:
        obs.archive.create_dataset('wavelengths', data=wl)

    if planet in obs.archive:
        print('deleting old copy')
        del obs.archive[planet]
    group = obs.archive.create_group(planet)

    u1, u2 = transit_params(planet).u
    duration = transit_duration(transit_params(planet))

    spectrum_photo = IRTFTemplate(sptype_phot)
    spectrum_spots = IRTFTemplate(sptype_spot)#spectrum_photo.scale_temperature(delta_teff)
    spectrum_secondary = IRTFTemplate(secondary_sptype)

    for midtransit in transits["{0} {1}".format(name, planet)]:
        print('midtransit', midtransit)
        times = np.arange(midtransit - 1.*duration,
                          midtransit + 1.*duration, exptime.to(u.day).value)
        # times = np.arange(midtransit - 1*duration, midtransit, exptime.to(u.day).value)

        transit = all_transits(times)

        subgroup = group.create_group("{0}".format(Time(midtransit, format='jd').isot))
        star = Star.with_k296_spot_distribution()
        area = star.spotted_area(times)
        fluxes = star.fractional_flux(times)
        # flares = inject_microflares(wl, times)

        spitzer_var = spitzer_variability(times)[:, np.newaxis]

        # oot = trappist_out_of_transit(times)
        # planet_area = transit_params(planet).rp**2 * (1-trappist_out_of_transit(times).astype(int))
        spectrum_photo_flux = spectrum_photo.interp_flux(wl)
        spectrum_spots_flux = spectrum_spots.interp_flux(wl)
        spectrum_secondary_flux = spectrum_secondary.interp_flux(wl)

        combined_spectra = (dilution_factor * ((transit - area[:, np.newaxis]) *
                             spectrum_photo_flux + area[:, np.newaxis] *
                             spectrum_spots_flux) +
                            (1 - dilution_factor) * spectrum_secondary_flux)

        # import ipdb; ipdb.set_trace()

        spectra = poisson(n_photons(wl, combined_spectra, exptime, mag,
                                    spectrum_photo.header) *
                          throughput(wl)[np.newaxis, :]
                          + background(wl, exptime)[np.newaxis, :])

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
        #subgroup.create_dataset('flares', data=1 + flares, **dataset_kwargs)
        subgroup.create_dataset('spitzer_var', data=spitzer_var, **dataset_kwargs)
        subgroup.create_dataset('times', data=times, **dataset_kwargs)
    obs.archive.flush()