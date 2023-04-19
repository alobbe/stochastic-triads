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
