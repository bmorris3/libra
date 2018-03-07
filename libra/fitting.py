
import celerite
import numpy as np
from celerite import terms
from scipy.optimize import minimize
from celerite.modeling import Model
from libra import transit_model
from copy import deepcopy
import emcee

from libra import trappist1


def log_probability(params, y, gp):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(y) + lp


class MeanModel3Param(Model):
    parameter_names = ['amp', 'depth', 't0']

    def get_value(self, t):
        params = deepcopy(trappist1('b'))
        params.rp = self.depth**0.5
        params.t0 = self.t0
        return self.amp * transit_model(t, params)


class MeanModel2Param(Model):
    parameter_names = ['amp', 'depth']

    def get_value(self, t):
        params = deepcopy(trappist1('b'))
        params.rp = self.depth**0.5
        return self.amp * transit_model(t, params)


def fit_bandintegrated(x, y, yerr, parameter_bounds, initp_dict, threads=4):

    mean_model = MeanModel3Param(bounds=parameter_bounds, **initp_dict)

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

    ndim, nwalkers = len(initial_params), len(initial_params) * 2

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    threads=threads, args=(y, gp))

    print('Fit band-integrated light curve...')
    p0 = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
    p0, lp, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    sampler.run_mcmc(p0, 2000)

    # Remove burn in
    samples = sampler.chain[:, 1000:, :].reshape(-1, ndim)

    # Get GP parameters
    log_S0 = np.median(samples[:, 0])
    log_omega0 = np.median(samples[:, 1])

    return samples, log_S0, log_omega0


def fit_spectral_bin(x, y, yerr, log_w0, log_S0, parameter_bounds, initp_dict,
                     threads=4):

    mean_model = MeanModel2Param(bounds=parameter_bounds, **initp_dict)

    Q = 1.0 / np.sqrt(2.0)
    kernel = terms.SHOTerm(log_S0=log_S0, log_Q=np.log(Q),
                           log_omega0=log_w0)

    kernel.freeze_parameter("log_Q")
    kernel.freeze_parameter("log_S0")
    kernel.freeze_parameter("log_omega0")

    gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
    gp.compute(x, yerr)

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

    ndim, nwalkers = len(initial_params), len(initial_params) * 2

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    threads=threads, args=(y, gp))

    p0 = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
    print('Fit spectral bin...')
    p0, lp, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    sampler.run_mcmc(p0, 1000)

    # Remove burn in
    samples = sampler.chain[:, 500:, :].reshape(-1, ndim)

    gp.set_parameter_vector(np.median(samples, axis=0))
    mu, var = gp.predict(y, x, return_var=True)

    # corner(samples, labels=['amp', 'depth'])
    # plt.show()
    return samples, mu
