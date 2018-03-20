import matplotlib.pyplot as plt
import numpy as np

from libra import (ObservationArchive, mask_simultaneous_transits_trappist,
                   transit_model, trappist1, nirspec_pixel_wavelengths)

from libra.fitting import fit_bandintegrated, fit_spectral_bin
from celerite.solver import LinAlgError

def model(params, planet, times):
    original_params = trappist1(planet)
    amp, depth, t0, c0, c1 = params
    original_params.rp = depth**0.5
    original_params.t0 = t0

    t = times - times.mean()

    fluxes = (amp * transit_model(times, original_params) +
              c0 * t +  c1 * t**2)

    return fluxes


def chi2(params, planet, times, observed_fluxes, obs_err):
    model_fluxes = model(params, planet, times)
    return np.sum( (observed_fluxes - model_fluxes)**2 / obs_err**2)


original_params = trappist1('b')

wavelengths = nirspec_pixel_wavelengths()

n_bins = 4
bin_width = len(wavelengths) // n_bins
bin_centers = np.array([wavelengths[i*bin_width:(i+1)*bin_width].mean().value
                        for i in range(n_bins)])

run_name = 'trappist1_bright2'

bin_results = []
bin_results_errs = []

import sys
planet = sys.argv[1]

with ObservationArchive(run_name + '_' + planet, 'r') as obs:
    for obs_planet in getattr(obs, planet):
        results = []
        errs = []
        mask = mask_simultaneous_transits_trappist(obs_planet.times, planet)

        obs_time = obs_planet.times[mask]
        obs_flux = np.sum(obs_planet.spectra[mask], axis=1)
        obs_err = np.sqrt(obs_flux)

        initp_dict = dict(amp=np.median(obs_flux), depth=original_params.rp**2,
                          t0=original_params.t0)

        parameter_bounds = dict(amp=[0.9*np.min(obs_flux), 1.3*np.max(obs_flux)],
                                depth=[0.9 * original_params.rp**2,
                                       1.1 * original_params.rp**2],
                                t0=[original_params.t0 - 0.05,
                                    original_params.t0 + 0.05])

        samples, log_S0, log_omega0 = fit_bandintegrated(obs_time, obs_flux, obs_err,
                                                         parameter_bounds,
                                                         initp_dict, threads=8)
        midtransit_posterior = samples[:, -1]

        l, mid, u = np.percentile(midtransit_posterior, [16, 50, 84])
        np.savetxt('outputs/time_{0}.txt'.format(obs_planet.attrs['t0']),
                   [mid, u-mid, mid-l])

        plt.hist(midtransit_posterior)
        plt.savefig("plots/midtransit_{0}.png".format(obs_planet.attrs['t0']))
        plt.close()

        depths = []
        depths_errs = []
        fig, ax = plt.subplots(1, n_bins, figsize=(14, 4))
        for i in range(n_bins):
            obs_time = obs_planet.times[mask]
            obs_flux = np.sum(obs_planet.spectra[mask, i*bin_width:(i+1)*bin_width], axis=1)
            obs_err = np.sqrt(obs_flux)

            initp_dict = dict(amp=np.median(obs_flux),
                              depth=original_params.rp**2)
            parameter_bounds = dict(amp=[0.9*np.min(obs_flux),
                                         1.3*np.max(obs_flux)],
                                    depth=[0.9 * original_params.rp**2,
                                           1.1 * original_params.rp**2])

            samples, mu = fit_spectral_bin(obs_time, obs_flux, obs_err,
                                           log_omega0, log_S0, parameter_bounds,
                                           initp_dict, threads=8)

            ax[i].plot(obs_time, obs_flux, '.')
            ax[i].plot(obs_time, mu, '.')

            depths.append(np.median(samples[:, -1]))
            depths_errs.append(np.std(samples[:, -1]))

        fig.tight_layout()
        fig.savefig("plots/fits_{0}_{1}.png".format(obs_planet.attrs['t0'], planet))
        plt.show()

        np.savetxt('outputs/transit_{0}_{1}.txt'.format(obs_planet.attrs['t0'], planet),
                   np.vstack([bin_centers, depths, depths_errs]).T)

        bin_results.append(depths)
        bin_results_errs.append(depths_errs)

            ## Begin celerite model
            # initp_dict = dict(amp=np.median(obs_flux), depth=original_params.rp**2,
            #                   t0=original_params.t0)
            #
            # parameter_bounds = dict(amp=[0.9*np.min(obs_flux), 1.3*np.max(obs_flux)],
            #                         depth=[0.9 * original_params.rp**2,
            #                                1.1 * original_params.rp**2],
            #                         t0=[original_params.t0 - 0.05,
            #                             original_params.t0 + 0.05])

            # try:

            #
            #
            # ax[i].plot(obs_time, obs_flux, '.')
            #
            #
            # try:
            #     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
            #                                     threads=4)
            #
            #     print("Running burn-in...")
            #     p0 = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
            #     p0, lp, _ = sampler.run_mcmc(p0, 1000)
            #
            #     print("Running production...")
            #     sampler.reset()
            #     sampler.run_mcmc(p0, 2000)
        #
        #         results.append(np.median(sampler.flatchain, axis=0))
        #         errs.append(np.std(sampler.flatchain, axis=0))
        #
        #         skip = 10
        #         mu, var = gp.predict(obs_flux, obs_time[::skip],
        #                              return_var=True)
        #         std = np.sqrt(var)
        #
        #         ax[i].plot(obs_time[::skip], mu, '.')
        #
        #         fig2 = corner(sampler.chain[:, 500:, :].reshape(-1, ndim),
        #                labels=['log_S', 'log_omega', 'amp', 'depth', 't0'])
        #         fig2.savefig("plots/corner_{0}_{1}.png"
        #                     .format(obs_planet.attrs['t0'], i),
        #                     dpi=200, bbox_inches='tight')
        #
        #     except LinAlgError:
        #         print('LinAlgError passing')
        #         results.append(np.nan * np.ones(ndim)) # np.array(bin_results[-1])
        #         errs.append(np.nan * np.ones(ndim)) # np.array(bin_results_errs[-1])
        # bin_results.append(results)
        # bin_results_errs.append(errs)
        # fig.tight_layout()
        # mid_transit_answer = obs_planet.attrs['t0']
        # fig.savefig("plots/fits_{0}_{1}.png".format(obs_planet.attrs['t0'], i))
        # plt.close()

#
# depths = np.array([j[3] for i in bin_results for j in i]).reshape(-1, n_bins)
#
# depths_errs = np.array([j[3] for i in bin_results_errs for j in i]).reshape(-1, n_bins)
#
#
# swl, sf = np.loadtxt('libra/data/transmission/t1b_10bar_venus_clear.txt', unpack=True)
#
# u = trappist1('b').u
# ld_factor = 1 - u[0]/3 - u[1]/6
#
# fig, ax = plt.subplots(1)#, 2, figsize=(12, 5))
# ax.plot(swl, sf * ld_factor, label='with LD')
#
# try:
#     for i in range(n_bins):
#         ax.errorbar(bin_centers + (i - 3)*0.1, depths[i, :], depths_errs[i, :], fmt='.')
# except IndexError:
#     pass
# ax.set(xlim=[0.5, 5.6], xlabel='Wavelength', ylabel='Depth')
#
# # model_interp = np.interp(bin_centers, swl, sf)
#
# # chi2 = np.sum((depths - model_interp)**2, axis=1)
#
# # mean_fluxes = []
# # with ObservationArchive(run_name, 'r') as obs:
# #     planet = 'b'
# #     for obs_planet in getattr(obs, planet):
# #         mf = np.mean(obs_planet.flares)
# #         mean_fluxes.append(mf)
#
# #ax[1].plot(mean_fluxes, chi2, '.')
# ax.set(xlabel='Wavelength', ylabel='Depth')
# fig.savefig('transmission.png', bbox_inches='tight', dpi=200)
# plt.show()