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
from emcee.utils import MPIPool
import sys

planet = sys.argv[1]

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
original_params = trappist1(planet)

from mpi4py import MPI

with ObservationArchive(run_name, 'a', outputs_dir=output_dir,
                        comm=MPI.COMM_WORLD, driver='mpio') as obs:

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

        parameter_bounds = dict(amp=[0.9*np.min(obs_flux), 1.1*np.max(obs_flux)],
                                depth=[0.5 * original_params.rp**2,
                                       1.5 * original_params.rp**2],
                                t0=[original_params.t0 - 0.1,
                                    original_params.t0 + 0.1])

        mean_model = MeanModel(bounds=parameter_bounds, **initp_dict)

        x = obs_time
        y = obs_flux
        yerr = obs_err

        Q = 1.0 / np.sqrt(2.0)
        log_w0 = 5 #3.0
        log_S0 = 10

        log_cadence_min = None # np.log(2*np.pi/(2./24))
        log_cadence_max = np.log(2*np.pi/(0.25/24))

        bounds = dict(log_S0=(-15, 30), log_Q=(-15, 15),
                      log_omega0=(log_cadence_min, log_cadence_max))

        kernel = terms.SHOTerm(log_S0=log_S0, log_Q=np.log(Q),
                               log_omega0=log_w0, bounds=bounds)

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

        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        pool=pool)

        print("Running burn-in...")
        p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 5000)

        print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0, 1000)
        pool.close()

        samples_log_S0 = sampler.flatchain[:, 0]
        samples_log_omega0 = sampler.flatchain[:, 1]

        samples_amp = sampler.flatchain[:, 2]
        samples_depth = sampler.flatchain[:, 3]
        samples_t0 = sampler.flatchain[:, 4]

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
            dset4 = group.create_dataset("amp", data=samples_amp,
                                         compression='gzip')
            obs.archive.flush()
        except KeyError:
            print('KeyError passing: {0}'.format(obs_planet.path))
