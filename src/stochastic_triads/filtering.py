################################################################################
# Copyright (c) 2023 Alexander Lobbe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

# pylint: disable=invalid-name, too-many-arguments

"""Functionality for filtering and ensemble forecasting"""

from functools import partial

import jax
import jax.numpy as jnp

from .ensemble import  Ensemble, run_ensemble, branching, ess
from .models import run_model

@partial(jax.jit, static_argnums=(1,3,6,9))
def run_forecast_triad(init_ensemble: Ensemble, # pylint: disable=too-many-locals
                       signal_model, signal_init,
                       ens_model,
                       params, dt, n_steps, key, obs_std, filt_interval):
    """Run a forecast experiment for a triad model"""

    ensembles = jnp.zeros((init_ensemble.positions.shape[0], n_steps, 3), dtype=jnp.complex_)
    signals = jnp.zeros((n_steps, 3), dtype=jnp.complex_)
    observations = jnp.zeros((n_steps, 3))

    ensemble = init_ensemble
    signal = signal_init

    # inner loop
    def loop_body(i,val):
        ensemble, ensembles, signals, observations, signal, key = val
        key, filter_key = jax.random.split(key)

        ensemble, new_ens_paths, new_signal_paths, _, new_observation = filter_step(ensemble, signal_model, signal, ens_model, params, dt, filt_interval, filter_key, obs_std) # pylint: disable=line-too-long

        signal = new_signal_paths[-1, :]

        # the ensembles returned at the assimilation times are the forecast ensembles
        # i.e pre-filtered ensemble
        ensembles = jax.lax.dynamic_update_slice(ensembles,new_ens_paths[:,-1,:][:,None,:], (0,i,0)) # pylint: disable=line-too-long
        signals = jax.lax.dynamic_update_slice(signals, signal[None, :],(i,0))
        observations = jax.lax.dynamic_update_slice(observations, new_observation[None, :], (i,0))

        return ensemble, ensembles, signals, observations, signal, key

    ensemble, ensembles, signals, observations, signal, key = jax.lax.fori_loop(0, n_steps, loop_body, (ensemble, ensembles, signals, observations, signal, key)) # pylint: disable=line-too-long

    return ensembles, signals, observations

@partial(jax.jit, static_argnums=(1,3,6,9))
def run_filter_triad(init_ensemble: Ensemble, # pylint: disable=too-many-locals
                     signal_model, signal_init,
                     ens_model,
                     params, dt, n_steps, key, obs_std, filt_interval):
    """Run a filtering experiment for a triad model"""

    ensemble_paths = jnp.zeros((init_ensemble.positions.shape[0], n_steps*filt_interval+1, 3), dtype=jnp.complex_) # pylint: disable=line-too-long
    signal_path = jnp.zeros((n_steps*filt_interval+1, 3), dtype=jnp.complex_)
    ess_s = jnp.zeros(n_steps)
    observations = jnp.zeros((n_steps, 3))

    ensemble = init_ensemble
    signal = signal_init

    def loop_body(i, val):
        ensemble, ensemble_paths, signal_path, ess_s, observations, signal, key = val
        key, filter_key = jax.random.split(key)

        ensemble, new_ens_paths, new_signal_paths, new_ess, new_observation = filter_step(ensemble, signal_model, signal, ens_model, params, dt, filt_interval, filter_key, obs_std) # pylint: disable=line-too-long

        # the ensembles returned at the assimilation times are the filtered ensembles
        ensemble_paths = jax.lax.dynamic_update_slice(ensemble_paths, new_ens_paths[:, :-1, :], (0, i*filt_interval, 0)) # pylint: disable=line-too-long
        signal_path = jax.lax.dynamic_update_slice(signal_path, new_signal_paths[:-1, :], (i*filt_interval, 0)) # pylint: disable=line-too-long
        ess_s = ess_s.at[i].set(new_ess)
        observations = jax.lax.dynamic_update_slice(observations, new_observation[None,:], (i, 0))

        signal = new_signal_paths[-1, :]

        return ensemble, ensemble_paths, signal_path, ess_s, observations, signal, key

    ensemble, ensemble_paths, signal_path, ess_s, observations, signal, key = jax.lax.fori_loop(0, n_steps, loop_body, (ensemble, ensemble_paths, signal_path, ess_s, observations, signal, key)) # pylint: disable=line-too-long

    ensemble_paths = ensemble_paths.at[:, -1, :].set(ensemble.positions)
    signal_path = signal_path.at[-1, :].set(signal)

    return ensemble_paths, signal_path, ess_s, observations

################################################################################
################################################################################

@partial(jax.jit, static_argnums=(1,3,6))
def filter_step(ensemble: Ensemble, # pylint: disable=too-many-locals
                signal_model, signal_init,
                ens_model,
                params, dt, filt_interval, key, obs_std):
    """Run a single step of the particle filter"""

    ens_key, signal_key, obs_key, branching_key = jax.random.split(key, 4)

    new_ens = run_ensemble(ens_model, params, dt, ensemble.positions, filt_interval, ens_key)

    new_signal = run_model(signal_model, params, dt, signal_init, filt_interval, signal_key)

    observation = generate_observation(new_signal[-1, :], obs_std, obs_key)
    weights = likelihood(new_ens[:, -1, :], observation, obs_std)
    ensemble_ess = ess(weights)
    weights = weights / jnp.sum(weights)

    pos, wgt, _ = branching(new_ens[:, -1, :], weights, branching_key)

    ens_out = Ensemble(pos, wgt, ensemble.n_ens)

    return ens_out, new_ens, new_signal, ensemble_ess, observation

################################################################################
################################################################################

def triad_sensor(x):
    """The sensor function for the triad model"""

    # We observe the modal energies of the signal
    return (x * jnp.conjugate(x)).real

def likelihood(pos, obs, obs_std):
    """Likelihood function"""

    return jnp.exp(log_likelihood(pos, obs, obs_std))

def log_likelihood(pos, obs, obs_std):
    """Log likelihood function"""

    dif = jnp.square(triad_sensor(pos) - obs) / obs_std**2
    return -0.5 * jnp.sum(dif, axis=1)

def generate_observation(signal, obs_std, key):
    """Generate an observation"""

    # The measurement is perturbed by gaussian noise
    return triad_sensor(signal) + jax.random.normal(key, shape=obs_std.shape) * obs_std

################################################################################
################################################################################

################################################################################
# DEPRECATED Use run_filter_triad instead
################################################################################
def _run_filter_triad_(init_ensemble: Ensemble, signal_model, signal_init, ens_model, params, dt, n_steps, key, obs_std, filt_interval):

    ensemble_paths = jnp.zeros((init_ensemble.positions.shape[0], n_steps*filt_interval+1, 3), dtype=jnp.complex_)
    signal_path = jnp.zeros((n_steps*filt_interval+1, 3), dtype=jnp.complex_)
    ess_s = jnp.zeros(n_steps)
    observations = jnp.zeros((n_steps, 3))

    ensemble = init_ensemble
    signal = signal_init
    for i in range(n_steps):

        key, filter_key = jax.random.split(key)

        ensemble, new_ens_paths, new_signal_paths, new_ess, new_observation = filter_step(ensemble, signal_model, signal, ens_model, params, dt, filt_interval, filter_key, obs_std)

        ensemble_paths = ensemble_paths.at[:, i*filt_interval:(i+1)*filt_interval, :].set(new_ens_paths[:, :-1, :])
        
        signal_path = signal_path.at[i*filt_interval:(i+1)*filt_interval, :].set(new_signal_paths[:-1, :])
        
        ess_s = ess_s.at[i].set(new_ess)

        observations = observations.at[i,:].set(new_observation)
        
        signal = new_signal_paths[-1, :]

    ensemble_paths = ensemble_paths.at[:, -1, :].set(ensemble.positions)
    signal_path = signal_path.at[-1, :].set(new_signal_paths[-1, :])    

    return ensemble_paths, signal_path, ess_s, observations

################################################################################
# DEPRECATED Use run_forecast_triad instead
################################################################################
def _run_forecast_triad_(init_ensemble: Ensemble, signal_model, signal_init, ens_model, params, dt, n_steps, key, obs_std, filt_interval):

    ensembles = jnp.zeros((init_ensemble.n_ens, n_steps, 3), dtype=jnp.complex_)
    signals = jnp.zeros((n_steps, 3), dtype=jnp.complex_)
    observations = jnp.zeros((n_steps, 3))

    ensemble = init_ensemble
    signal = signal_init

    for i in range(n_steps):

        key, filter_key = jax.random.split(key)

        ensemble, new_ens_paths, new_signal_paths, _, new_observation = filter_step(ensemble, signal_model, signal, ens_model, params, dt, filt_interval, filter_key, obs_std)

        signal = new_signal_paths[-1, :]

        ensembles = ensembles.at[:, i, :].set(new_ens_paths[:,-1,:])
        signals = signals.at[i, :].set(signal)
        observations = observations.at[i,:].set(new_observation)  

    return ensembles, signals, observations