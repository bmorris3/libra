import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import celerite
from celerite import terms
from scipy.optimize import minimize
from celerite.modeling import Model
from copy import deepcopy
from astropy.utils.console import ProgressBar

from libra import (trappist1, transit_model, ObservationArchive,
                   trappist_out_of_transit, nirspec_pixel_wavelengths,
                   transit_duration)

t0 = trappist1('b').t0
original_params = trappist1('b')
duration = transit_duration(original_params)
times = np.arange(t0-1.5*duration, t0+1.5*duration, 1/60/60/24)

import sys
j = sys.argv[1]

outputs_dir = '/astro/store/scratch/tmp/bmmorris/libra/outputs'

with ObservationArchive('trappist1_bright2_b', 'r', outputs_dir=outputs_dir) as obs:
# with ObservationArchive('trappist1_bright2_b', 'r') as obs:
    sim = obs.b[int(j)]
    times = sim.times[:]
    spectra = sim.spectra[:]
    plt.plot(sim.times, np.sum(sim.spectra, axis=1))

oot = trappist_out_of_transit(times)

fluxes = np.sum(spectra, axis=1)
oot_median = np.median(fluxes[oot])
errors = np.sqrt(fluxes) / oot_median
fluxes /= oot_median
fluxes -= 1

plt.errorbar(times, fluxes, errors, color='k', fmt='.', ecolor='silver')

class MeanModel3Param(Model):
    parameter_names = ['amp', 'depth', 't0']

    def get_value(self, t):
        params = deepcopy(trappist1('b'))
        params.rp = self.depth**0.5
        params.t0 = self.t0 + original_params.t0
        return self.amp * transit_model(t, params)

initp_dict = dict(amp=np.median(fluxes), depth=original_params.rp**2,
                  t0=0)#t0=original_params.t0)

parameter_bounds = dict(amp=[0.9*np.min(fluxes), 1.3*np.max(fluxes)],
                        depth=[0.9 * original_params.rp**2,
                               1.1 * original_params.rp**2],
                        t0=[-0.05, 0.05])

mean_model = MeanModel3Param(bounds=parameter_bounds, **initp_dict)

Q = 1.0 / np.sqrt(2.0)
log_w0 = 4 #3.0
log_S0 = 10

log_cadence_min = None # np.log(2*np.pi/(2./24))
log_cadence_max = np.log(2*np.pi/(0.25/24))

bounds = dict(log_S0=(-15, 30), log_Q=(-15, 15),
              log_omega0=(log_cadence_min, log_cadence_max))

kernel = terms.SHOTerm(log_S0=log_S0, log_Q=np.log(Q),
                       log_omega0=log_w0, bounds=bounds)

kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
gp.compute(times - original_params.t0, errors)

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
                method="L-BFGS-B", bounds=bounds, args=(fluxes, gp))
gp.set_parameter_vector(soln.x)

mu, var = gp.predict(fluxes, times - original_params.t0, return_var=True)
std = np.sqrt(var)

fixed_transit_model = mu.copy()

betas = np.zeros(spectra.shape[1])
betas_errs = np.zeros(spectra.shape[1])

with ProgressBar(len(betas)) as bar:
    for i in range(spectra.shape[1]):
        single_channel_flux = spectra[:, i].copy()
        single_channel_error = np.sqrt(single_channel_flux)

        X = fixed_transit_model[:, np.newaxis]# - 1
        y = single_channel_flux / np.median(single_channel_flux[oot]) - 1
        err = single_channel_error / np.median(single_channel_flux[oot])

        if not np.isnan(y).any():
            omega = np.diag(err**2) #+ cov
            omega_inv = np.linalg.inv(omega)

            V = np.linalg.inv(X.T @ omega_inv @ X)
            beta = V @ X.T @ omega_inv @ y
            betas[i] = beta[0]
            betas_errs[i] = np.sqrt(np.diag(V))

        else:
            betas[i] = np.nan
            betas_errs[i] = np.nan
        bar.update()

rp = trappist1('b').rp
depth = rp**2
u = trappist1('b').u
ld_factor = 1 - u[0]/3 - u[1]/6
wl = nirspec_pixel_wavelengths()

np.savetxt('outputs/transmission_spectrum_b_{0:03d}.txt'.format(j),
           np.vstack([wl, betas * depth, betas_errs * depth]).T)

#plt.errorbar(wl.value, betas * depth, betas_errs * depth