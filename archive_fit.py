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

from celerite.modeling import Model
from copy import deepcopy
import emcee




class MeanModel(Model):
    parameter_names = ['amp', 'depth', 't0']

    def get_value(self, t):
        params = deepcopy(original_params)
        params.rp = self.depth**0.5
        params.t0 = self.t0
        return self.amp * transit_model(t, params)


run_name = 'trappist1_bright'
output_dir = '/gscratch/stf/bmmorris/libra/'

#with ObservationArchive(run_name, 'a', outputs_dir=output_dir) as obs:
for planet in list('h'): #list('bcdefgh'):
    original_params = trappist1(planet)
    with ObservationArchive(run_name, 'a') as obs:

        for obs_planet in getattr(obs, planet):
            print(planet, obs_planet.path)
            mask = mask_simultaneous_transits(obs_planet.times, planet)
            obs_time = obs_planet.times[mask]
            obs_flux = np.sum(obs_planet.spectra[mask], axis=1)
            obs_err = np.sqrt(obs_flux)

            params = trappist1(planet)

            # plt.plot(obs_time, obs_flux, '.')

            mid_transit_answer = obs_planet.attrs['t0']
            # plt.show()

            initp_dict = dict(amp=np.median(obs_flux), depth=original_params.rp**2,
                              t0=original_params.t0)

            parameter_bounds = dict(amp=[np.min(obs_flux), np.max(obs_flux)],
                                    depth=[0.5 * original_params.rp**2,
                                           1.5 * original_params.rp**2],
                                    t0=[original_params.t0 - 0.1,
                                        original_params.t0 + 0.1])

            mean_model = MeanModel(**initp_dict, bounds=parameter_bounds)

            x = obs_time
            y = obs_flux
            yerr = obs_err/2

            Q = 1.0 / np.sqrt(2.0)
            w0 = 3.0
            S0 = 10
            log_cadence = np.log(1/60/60/24)

            bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15),
                          log_omega0=(log_cadence, 15))
            kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q),
                                   log_omega0=np.log(w0), bounds=bounds)

            kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

            bounds = dict(log_sigma=(-15, 15), log_rho=(log_cadence, 100))
            kernel += terms.Matern32Term(log_sigma=0, log_rho=0, bounds=bounds)

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
            print("Final log-likelihood: {0}".format(-soln.fun))

            skip = 10
            mu, var = gp.predict(obs_flux, obs_time[::skip], return_var=True)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(obs_time, obs_flux)
            # plt.plot(obs_time[::skip], mu)
            # plt.show()
            def log_probability(params):
                gp.set_parameter_vector(params)
                lp = gp.log_prior()
                if not np.isfinite(lp):
                    return -np.inf
                return gp.log_likelihood(y) + lp

            initial = np.array(soln.x)
            ndim, nwalkers = len(initial), len(initial) * 2

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            threads=4)

            print("Running burn-in...")
            p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
            #p0, lp, _ = sampler.run_mcmc(p0, 10000)
            p0, lp, _ = sampler.run_mcmc(p0, 2000)

            print("Running production...")
            sampler.reset()
            sampler.run_mcmc(p0, 1000)


            # from corner import corner
            #
            # corner(sampler.flatchain)
            # plt.show()

            samples_log_S0 = sampler.flatchain[:, 0]
            samples_log_omega0 = sampler.flatchain[:, 1]
            samples_log_sigma = sampler.flatchain[:, 2]
            samples_log_rho = sampler.flatchain[:, 3]

            samples_amp = sampler.flatchain[:, 4]
            samples_depth = sampler.flatchain[:, 5]
            samples_t0 = sampler.flatchain[:, 6]
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
                if "log_S0" in group: 
                    del group["log_S0"]
                if "log_omega0" in group: 
                    del group["log_omega0"]
                if "log_sigma" in group:
                    del group["log_sigma"]
                if "log_rho" in group:
                    del group["log_rho"]

            dset0 = group.create_dataset("depth", data=samples_depth,
                                         compression='gzip')
            dset1 = group.create_dataset("t0", data=samples_t0,
                                         compression='gzip')
            dset2 = group.create_dataset("log_S0", data=samples_log_S0,
                                         compression='gzip')
            dset3 = group.create_dataset("log_omega0", data=samples_log_omega0,
                                         compression='gzip')
            dset4 = group.create_dataset("amp", data=samples_amp,
                                         compression='gzip')
            dset5 = group.create_dataset("log_sigma", data=samples_log_sigma,
                                         compression='gzip')
            dset6 = group.create_dataset("log_rho", data=samples_log_rho,
                                         compression='gzip')
            obs.archive.flush()
