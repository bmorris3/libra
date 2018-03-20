"""
Run with

mpirun -np 8 python archive_fit.py

"""
import numpy as np
import celerite
from celerite import terms
from scipy.optimize import minimize
from libra import (ObservationArchive, mask_simultaneous_transits_k296,
                   mask_simultaneous_transits_trappist,
                   transit_model, kepler296, trappist1, kepler62,
                   mask_simultaneous_transits_k62)
from celerite.solver import LinAlgError
from celerite.modeling import Model
from copy import deepcopy
import emcee
import sys


planet = sys.argv[1]

run_name = 'k62_ngroups45_10transits'
original_params = kepler62(planet)
simultaenous_transits = mask_simultaneous_transits_k62

# run_name = 'k296'
# original_params = kepler296(planet)
# simultaenous_transits = mask_simultaneous_transits_k296
#
# run_name = 'trappist1_ngroups2' #'trappist1_bright2' #'trappist1_microflares' #'trappist1_bright'
# original_params = trappist1(planet)
# simultaenous_transits = mask_simultaneous_transits_trappist

with ObservationArchive(run_name + '_' + planet, 'a') as obs:
    #for obs_planet in getattr(obs, planet):
    n_iterations = len(getattr(obs, planet))

n_iterations = np.min([n_iterations, 10])

for i in range(0, n_iterations):
    with ObservationArchive(run_name + '_' + planet, 'a') as obs:
        #for obs_planet in getattr(obs, planet):
        obs_planet = getattr(obs, planet)[i]
        print(planet, obs_planet.path)
        mask = simultaenous_transits(obs_planet.times, planet)
        #not_during_flares = np.abs(np.max(obs_planet.flares, axis=1) - 1) < 1e-3
        #mask &= not_during_flares

        mid_transit_answer = obs_planet.attrs['t0']

        obs_time = obs_planet.times[mask] - mid_transit_answer #- original_params.t0
        obs_flux = np.sum(obs_planet.spectra[mask], axis=1)
        obs_err = np.sqrt(obs_flux)

        obs_err /= np.median(obs_flux)
        obs_flux /= np.median(obs_flux)
        #
        # import matplotlib.pyplot as plt
        # print(obs_time, obs_flux)
        # plt.plot(obs_time, obs_flux)
        # plt.show()

        #params = trappist1(planet)

        class MeanModel(Model):
            parameter_names = ['amp', 'depth', 't0']

            def get_value(self, t):
                params = deepcopy(original_params)
                params.rp = self.depth**0.5
                params.t0 = self.t0 + mid_transit_answer
                return self.amp * transit_model(t + mid_transit_answer, params)

        initp_dict = dict(amp=np.median(obs_flux), depth=original_params.rp**2,
                          t0=0)

        parameter_bounds = dict(amp=[np.min(obs_flux), np.max(obs_flux)],
                                depth=[0.5 * original_params.rp**2,
                                       1.5 * original_params.rp**2],
                                t0=[-0.001, 0.001])
                                # t0=[mid_transit_answer-0.01, mid_transit_answer+0.01])
                                # t0=[original_params.t0-0.1, original_params.t0+0.1])

        mean_model = MeanModel(bounds=parameter_bounds, **initp_dict)


        # import matplotlib.pyplot as plt
        # plt.plot(obs_time, obs_flux, '.')
        # plt.plot(obs_time, mean_model.get_value(obs_time))
        # plt.show()

#        x = obs_time
#        y = obs_flux
#        yerr = obs_err
        y = obs_flux

        Q = 1.0 / np.sqrt(2.0)
        log_w0 = np.log(2*np.pi/(8/24))
        log_S0 = -15

        log_cadence_min = None # np.log(2*np.pi/(2./24))
        log_cadence_max = np.log(2*np.pi/(6/24))#np.log(2*np.pi/(0.25/24))

        #bounds = dict(log_S0=(-10, 10), log_Q=(-15, 15),
        bounds = dict(log_S0=(-30, 15), log_Q=(-15, 15),
                      log_omega0=(log_cadence_min, log_cadence_max))

        kernel = terms.SHOTerm(log_S0=log_S0, log_Q=np.log(Q),
                               log_omega0=log_w0, bounds=bounds)

        kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

        gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
        gp.compute(obs_time, obs_err)
        print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))

        # from interpacf import interpolated_acf
        # import matplotlib.pyplot as plt
        # lag, acf = interpolated_acf(x, y-np.median(y))
        # plt.plot(lag, acf/acf.max())
        # plt.plot(lag, kernel.get_value(lag)/kernel.get_value(lag).max())
        # plt.show()

        # Define a cost function
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)

        def grad_neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.grad_log_likelihood(y)[1]

        # Fit for the maximum likelihood parameters
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(neg_log_like, initial_params, #jac=grad_neg_log_like,
                        method="L-BFGS-B", bounds=bounds, args=(y, gp))
        gp.set_parameter_vector(soln.x)
        print(soln.x)
        print("Final log-likelihood: {0}".format(-soln.fun))

        def log_prior(params):
            #log_a, log_c, log_d, log_omega0, log_S0, log_Q, amp, depth, t0 = params
            log_omegaS0, log_omega0, log_Q, amp, depth, t0 = params
            #log_a, amp, depth, t0 = params

            if not ((parameter_bounds['amp'][0] < amp < parameter_bounds['amp'][1])
                    and (parameter_bounds['depth'][0] < depth < parameter_bounds['depth'][1])
                    and (parameter_bounds['t0'][0] < t0 < parameter_bounds['t0'][1])):
                return -np.inf

            return 0

        def log_probability(params):
            gp.set_parameter_vector(params)
            lp = gp.log_prior()
            # lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return gp.log_likelihood(y) + lp

        initial = np.array(soln.x)
        # initial = initial_params = gp.get_parameter_vector()
        ndim, nwalkers = len(initial), len(initial) * 4
        #
        # import matplotlib.pyplot as plt
        # plt.errorbar(obs_time, obs_flux, obs_err, fmt='.', ecolor='gray')
        # gp.set_parameter_vector(initial_params)
        # gp.compute(obs_time, obs_err)
        # mu, var = gp.predict(obs_flux, obs_time, return_var=True)
        # plt.plot(obs_time, mu, color='C1', zorder=10)
        #
        # #gp.set_parameter_vector(soln.x)
        # # mu, var = gp.predict(obs_flux, obs_time, return_var=True)
        # # plt.plot(obs_time, mu, color='C2')
        # plt.show()

        try:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            threads=8) #pool=pool)

            print("Running burn-in...")
            p0 = initial + 1e-6 * np.random.randn(nwalkers, ndim)
            p0, lp, _ = sampler.run_mcmc(p0, 1000)
            # p0, lp, _ = sampler.run_mcmc(p0, 2000)
            sampler.reset()
            # sampler.run_mcmc(p0, 5000)
            p0, lp, _ = sampler.run_mcmc(p0, 1000)
            print("Running production...")
            sampler.reset()
            sampler.run_mcmc(p0, 2000)


            #pool.close()

            # from corner import corner
            # import matplotlib.pyplot as plt
            # corner(sampler.flatchain, labels='log_S0 log_omega0 amp depth t0'.split())
            # plt.savefig('plots/corner_{0}_{1}.png'.format(i, planet), bbox_inches='tight', dpi=200)
            # plt.close()

            samples_log_S0 = sampler.flatchain[:, 0]
            samples_log_omega0 = sampler.flatchain[:, 1]
            # samples_log_a = sampler.flatchain[:, 0]
            samples_amp = sampler.flatchain[:, 2]
            samples_depth = sampler.flatchain[:, 3]
            samples_t0 = sampler.flatchain[:, 4]

            if not 'samples' in obs.archive[obs_planet.path]:
                group = obs.archive[obs_planet.path].create_group('samples')

            else:
                group = obs.archive[obs_planet.path + 'samples']
                if "depth" in group:
                    del group["depth"]
                if 't0' in group:
                    del group["t0"]
                if "amp" in group:
                    del group["amp"]
                # if "log_a" in group:
                #     del group["log_a"]
                if "log_S0" in group:
                    del group["log_S0"]
                if "log_omega0" in group:
                    del group["log_omega0"]

            dset0 = group.create_dataset("depth", data=samples_depth,
                                         compression='gzip')
            dset1 = group.create_dataset("t0", data=samples_t0,
                                         compression='gzip')
            dset2 = group.create_dataset("log_S0", data=samples_log_S0,
                                         compression='gzip')
            dset3 = group.create_dataset("log_omega0", data=samples_log_omega0,
                                         compression='gzip')
            # dset2 = group.create_dataset("log_a", data=samples_log_a,
            #                              compression='gzip')
            dset3 = group.create_dataset("amp", data=samples_amp,
                                         compression='gzip')
            obs.archive.flush()
        except (KeyError, LinAlgError) as e:
            print('Error {0}. passing: {1}'.format(e, obs_planet.path))
