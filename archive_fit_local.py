"""
Run with

mpirun -np 8 python archive_fit.py

"""
import numpy as np
import celerite
from celerite import terms
from scipy.optimize import minimize
from libra import (ObservationArchive, mask_simultaneous_transits,
                   transit_model, trappist1)
from celerite.solver import LinAlgError
from celerite.modeling import Model
from copy import deepcopy
import emcee
import sys

planet = sys.argv[1]

run_name = 'trappist1_bright2' #'trappist1_microflares' #'trappist1_bright'

original_params = trappist1(planet)


class MeanModel(Model):
    parameter_names = ['amp', 'depth', 't0']

    def get_value(self, t):
        params = deepcopy(original_params)
        params.rp = self.depth**0.5
        params.t0 = self.t0
        return self.amp * transit_model(t, params)


with ObservationArchive(run_name + '_' + planet, 'a') as obs:

    for obs_planet in getattr(obs, planet)[:10]:
        print(planet, obs_planet.path)
        mask = mask_simultaneous_transits(obs_planet.times, planet)
        #not_during_flares = np.abs(np.max(obs_planet.flares, axis=1) - 1) < 1e-3
        #mask &= not_during_flares

        obs_time = obs_planet.times[mask]
        obs_flux = np.sum(obs_planet.spectra[mask], axis=1)
        obs_err = np.sqrt(obs_flux)

        #params = trappist1(planet)

        mid_transit_answer = obs_planet.attrs['t0']
        # import matplotlib.pyplot as plt
        # plt.plot(obs_time, obs_flux, '.')
        # plt.show()

        initp_dict = dict(amp=np.median(obs_flux), depth=original_params.rp**2,
                          t0=original_params.t0)

        parameter_bounds = dict(amp=[0.9*np.min(obs_flux), 1.1*np.max(obs_flux)],
                                depth=[0.9 * original_params.rp**2,
                                       1.1 * original_params.rp**2],
                                t0=[original_params.t0 - 0.01,
                                    original_params.t0 + 0.01])

        mean_model = MeanModel(bounds=parameter_bounds, **initp_dict)

        x = obs_time
        y = obs_flux
        yerr = obs_err
        #
        # Q = 1.0 / np.sqrt(2.0)
        # log_w0 = 5 #3.0
        # log_S0 = 10
        #
        # log_cadence_min = None # np.log(2*np.pi/(2./24))
        # log_cadence_max = np.log(2*np.pi/(0.25/24))
        #
        # bounds = dict(log_S0=(-15, 30), log_Q=(-15, 15),
        #               log_omega0=(log_cadence_min, log_cadence_max))
        #
        # kernel = terms.SHOTerm(log_S0=log_S0, log_Q=np.log(Q),
        #                        log_omega0=log_w0, bounds=bounds)

        # kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

        log_a_median = -14.49285716
        log_c_median = 1.98108915

        kernel = terms.RealTerm(log_a=log_a_median, log_c=log_c_median)#, bounds=bounds)

        # kernel.freeze_parameter('log_c')

        gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
        gp.compute(x, yerr)
        print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))

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
            log_a, amp, depth, t0 = params

            if not ((parameter_bounds['amp'][0] < amp < parameter_bounds['amp'][1])
                    and (parameter_bounds['depth'][0] < depth < parameter_bounds['depth'][1])
                    and (parameter_bounds['t0'][0] < t0 < parameter_bounds['t0'][1])):
                return -np.inf

            # Add priors for GP parameters
            #prior_penalty = -0.5 * (log_c - log_c_median)**2 / 0.15**2

            return 0 #prior_penalty

        def log_probability(params):
            gp.set_parameter_vector(params)
            #lp = gp.log_prior()
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return gp.log_likelihood(y) + lp

        initial = np.array(soln.x)
        # initial = initial_params = gp.get_parameter_vector()
        ndim, nwalkers = len(initial), len(initial) * 2

        import matplotlib.pyplot as plt
        plt.scatter(obs_time, obs_flux)
        gp.set_parameter_vector(initial_params)
        gp.compute(obs_time, obs_err)
        mu, var = gp.predict(obs_flux, obs_time, return_var=True)
        plt.plot(obs_time, mu, color='C1')

        #gp.set_parameter_vector(soln.x)
        # mu, var = gp.predict(obs_flux, obs_time, return_var=True)
        # plt.plot(obs_time, mu, color='C2')
        plt.show()

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        threads=8) #pool=pool)

        print("Running burn-in...")
        p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 8000)

        print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0, 2000)
        #pool.close()

        # from corner import corner
        #
        # corner(sampler.flatchain, labels='log_a log_c amp depth t0'.split())
        # plt.show()

        # samples_log_S0 = sampler.flatchain[:, 0]
        # samples_log_omega0 = sampler.flatchain[:, 1]
        samples_log_a = sampler.flatchain[:, 0]
        samples_amp = sampler.flatchain[:, 1]
        samples_depth = sampler.flatchain[:, 2]
        samples_t0 = sampler.flatchain[:, 3]

        try:
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
                if "log_a" in group:
                    del group["log_a"]
                # if "log_S0" in group:
                #     del group["log_S0"]
                # if "log_omega0" in group:
                #     del group["log_omega0"]

            dset0 = group.create_dataset("depth", data=samples_depth,
                                         compression='gzip')
            dset1 = group.create_dataset("t0", data=samples_t0,
                                         compression='gzip')
            # dset2 = group.create_dataset("log_S0", data=samples_log_S0,
            #                              compression='gzip')
            # dset3 = group.create_dataset("log_omega0", data=samples_log_omega0,
            #                              compression='gzip')
            dset2 = group.create_dataset("log_a", data=samples_log_a,
                                         compression='gzip')
            dset3 = group.create_dataset("amp", data=samples_amp,
                                         compression='gzip')
            obs.archive.flush()
        except (KeyError, LinAlgError) as e:
            print('Error {0}. passing: {1}'.format(e, obs_planet.path))
