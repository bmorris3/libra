import numpy as np
import celerite
from celerite import terms
from scipy.optimize import minimize

from libra import (ObservationArchive, mask_simultaneous_transits,
                   transit_model, trappist1)

from celerite.modeling import Model
from copy import deepcopy
import emcee

original_params = trappist1('h')


class MeanModel(Model):
    parameter_names = ['amp', 'depth', 't0']

    def get_value(self, t):
        params = deepcopy(original_params)
        params.rp = self.depth**0.5
        params.t0 = self.t0
        return self.amp * transit_model(t, params)


run_name = 'trappist1_bright'
output_dir = '/gscratch/stf/bmmorris/libra/'

with ObservationArchive(run_name, 'a', output_dir=output_dir) as obs:
    planet = 'h'
    for obs_planet in getattr(obs, planet):
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
                                t0=[original_params.t0 - 0.5,
                                    original_params.t0 + 0.5])

        mean_model = MeanModel(**initp_dict, bounds=parameter_bounds)

        x = obs_time
        y = obs_flux
        yerr = obs_err/2

        Q = 1.0 / np.sqrt(2.0)
        w0 = 3.0
        S0 = 10

        bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
        kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q),
                               log_omega0=np.log(w0), bounds=bounds)

        kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

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

        def log_probability(params):
            gp.set_parameter_vector(params)
            lp = gp.log_prior()
            if not np.isfinite(lp):
                return -np.inf
            return gp.log_likelihood(y) + lp

        initial = np.array(soln.x)
        ndim, nwalkers = len(initial), len(initial) * 2
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        threads=8)

        print("Running burn-in...")
        p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 10000)

        print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0, 5000)

        samples_depth = sampler.flatchain[:, -2]
        samples_t0 = sampler.flatchain[:, -1]

        group = obs.archive[obs_planet.path].create_group('samples')
        dset0 = group.create_dataset("depth", data=samples_depth, compression='gzip')
        dset1 = group.create_dataset("t0", data=samples_t0, compression='gzip')
        obs.archive.flush()
