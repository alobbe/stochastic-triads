################################################################################
# Copyright 2023 Alexander Lobbe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# pylint: disable=invalid-name, too-many-arguments

"""Functionality for ensemble simulations of triad models"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

from .models import run_model

class Ensemble(NamedTuple):
    """Ensemble state"""

    positions: jnp.ndarray
    weights: jnp.ndarray
    n_ens: int

@partial(jax.jit, static_argnums=(0,4))
def run_ensemble(model, params, dt, initial, steps, key):
    """Run an ensemble of a given triad model"""

    keys = jax.random.split(key, initial.shape[0])

    def _model_fun(initial, key):
        return run_model(model=model, params=params, dt=dt, steps=steps, initial=initial, key=key)

    # vectorized run of the model
    ensemble = jax.vmap(_model_fun, in_axes=(0,0), out_axes=0)(initial, keys)

    return ensemble

def ess(weights):
    """Compute the effective sample size"""

    return jnp.square(jnp.sum(weights)) / jnp.sum(jnp.square(weights))

@jax.jit
def branching(positions, weights, key):
    """Perform a branching step"""

    N_ens = len(weights)
    offspring = compute_offspring(weights, key)
    new_positions, parent = reindex(positions, offspring)
    new_weights = jnp.ones_like(weights) * (1./N_ens)

    return new_positions, new_weights, parent

################################################################################
################################################################################

@jax.jit
def compute_offspring(weights, key):
    """Compute the number of offspring for each particle"""

    def inner_cond_true(unif, offspr, j, n, weights, h, g):
        offspr = jax.lax.cond(unif[j] < 1. - (((n*weights[j]) % 1)/((g % 1)+1e-10)),
                    lambda offspr, j, n, weights, h, g: offspr.at[j].set(jnp.floor(n*weights[j])),
                    lambda offspr, j, n, weights, h, g: offspr.at[j].set(jnp.floor(n*weights[j]) + h - jnp.floor(g)), # pylint: disable=line-too-long
                    offspr, j, n, weights, h, g  )
        return offspr

    def inner_cond_false(unif, offspr, j, n, weights, h, g):
        offspr = jax.lax.cond(unif[j] < 1. - (1. - ((n*weights[j]) % 1) ) / (1 - (g % 1)+1e-10),
                    lambda offspr, j, n, weights, h, g: offspr.at[j].set(jnp.floor(n*weights[j]) + 1.), # pylint: disable=line-too-long
                    lambda offspr, j, n, weights, h, g: offspr.at[j].set(jnp.floor(n*weights[j]) + h - jnp.floor(g)), # pylint: disable=line-too-long
                    offspr, j, n, weights, h, g  )
        return offspr

    def outer_cond(unif, offspr, j, n, weights, h, g):
        offspr = jax.lax.cond(((n*weights[j]) % 1) + ((g - n*weights[j]) % 1) < 1.,
                    inner_cond_true,
                    inner_cond_false,
                    unif, offspr, j, n, weights, h, g  )
        return offspr

    def body_fun(i, val):
        unif, offspr, n, weights, h, g = val
        offspr = outer_cond(unif, offspr, i, n, weights, h, g)
        g = g - n*weights[i]
        h = h - offspr[i]
        return unif, offspr, n, weights, h, g

    n = weights.shape[0]
    unif = jax.random.uniform(key, shape=(n-1,))
    g = n
    h = n
    offspr = jnp.empty_like(weights)

    unif, offspr, n, weights, h, g = jax.lax.fori_loop(0, n-1, body_fun, (unif, offspr, n, weights, h, g)) # pylint: disable=line-too-long

    offspr = offspr.at[n-1].set(h)
    offspr = offspr.astype(int)
    return offspr

@jax.jit
def reindex(positions, offspring):
    """Reindex the particles according to their offspring"""

    def body_fun_inner(i, val):
        pos, parent, r, positions, j = val
        pos = pos.at[r+i].set(positions[j])
        parent = parent.at[r+i].set(j) # parent of particle r+i is particle j
        return pos, parent, r, positions, j

    def body_fun_outer(j, val):
        pos, parent, r, positions, offspring = val
        pos, parent, r, positions, _ = jax.lax.fori_loop(0, offspring[j], body_fun_inner, (pos, parent, r, positions, j)) # pylint: disable=line-too-long
        r = r + offspring[j]
        return pos, parent, r, positions, offspring

    n = len(positions)
    pos = jnp.empty_like(positions)
    parent = jnp.empty_like(offspring)
    r = 0
    pos, parent, r, positions, offspring = jax.lax.fori_loop(0, n, body_fun_outer, (pos, parent, r, positions, offspring)) # pylint: disable=line-too-long

    return pos, parent
