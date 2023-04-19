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

"""Triad models"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

class TriadParams(NamedTuple):
    """Parameters for triad models"""

    # Triad wave vectors
    k: jnp.ndarray
    p: jnp.ndarray
    q: jnp.ndarray

    # Parities
    s_k: float
    s_p: float
    s_q: float

    # Noise Amplitudes
    b_k: float
    b_p: float
    b_q: float

    # Geometric quantities
    gamma: jnp.ndarray
    g: float
    h_k: jnp.ndarray
    h_p: jnp.ndarray
    h_q: jnp.ndarray
    D: jnp.ndarray

def get_triad_params(k, p, q, s_k, s_p, s_q, b_k, b_p, b_q, gamma):
    """Get triad parameters

    Read in the parameters for the triad model and return a
    TriadParams object containing also the derived parameters.
    """

    # cast input parameters to jax arrays
    k = jnp.array(k)
    p = jnp.array(p)
    q = jnp.array(q)
    gamma = jnp.array(gamma)

    # compute derived parameters
    g = _g(k, p, q, s_k, s_p, s_q, gamma)
    h_k = _h(k, s_k, gamma)
    h_p = _h(p, s_p, gamma)
    h_q = _h(q, s_q, gamma)
    D = jnp.array([s_k*jnp.linalg.norm(k, ord=2),
                    s_p*jnp.linalg.norm(p, ord=2),
                    s_q*jnp.linalg.norm(q, ord=2)])

    return TriadParams(k, p, q,
                       s_k, s_p, s_q,
                       b_k, b_p, b_q,
                       gamma, g, h_k, h_p, h_q, D)

@partial(jax.jit, static_argnums=(0,4))
def run_model(model, params, dt, initial, steps, key):
    """Run a triad model"""

    if model == "HST":
        model_step = model_step_HST
    elif model == "EST":
        model_step = model_step_EST
    elif model == "DET":
        model_step = model_step_DET
    else:
        raise ValueError("Invalid model")

    # inner loop
    def _step(carry, i):
        y, key = carry
        y = model_step(params, dt, y, key)
        key = jax.random.fold_in(key, i)
        return (y, key), y

    _, ys = jax.lax.scan(_step, (initial, key), jnp.arange(steps))

    # prepend initial condition
    ys = jnp.concatenate([initial.reshape(1, -1), ys], axis=0)

    return ys

################################################################################
################################################################################

def model_step_HST(params, dt, initial, key):
    """Run one step of the HST model"""

    noise = jax.random.normal(key)

    # SSPRK3
    q_1 = (initial
           + dt * _triad_rhs(params, initial)
           + _triad_rhs_HST(params, initial) * jnp.sqrt(dt) * noise)
    q_2 = ((3/4) * initial
           + (1/4) * (q_1 + dt * _triad_rhs(params, q_1)
                      + _triad_rhs_HST(params, q_1) * jnp.sqrt(dt) * noise))
    y = ((1/3) * initial
         + (2/3) * (q_2 + dt * _triad_rhs(params, q_2)
                    + _triad_rhs_HST(params, q_2) * jnp.sqrt(dt) * noise))

    return y

def model_step_EST(params, dt, initial, key):
    """Run one step of the EST model"""

    noise = jax.random.normal(key)

    # SSPRK3
    q_1 = (initial
           + dt * _triad_rhs(params, initial)
           + _triad_rhs_EST(params, initial) * jnp.sqrt(dt) * noise)
    q_2 = ((3/4) * initial
           + (1/4) * (q_1 + dt * _triad_rhs(params, q_1)
                      + _triad_rhs_EST(params, q_1) * jnp.sqrt(dt) * noise))
    y = ((1/3) * initial
         + (2/3) * (q_2 + dt * _triad_rhs(params, q_2)
                    + _triad_rhs_EST(params, q_2) * jnp.sqrt(dt) * noise))

    return y

def model_step_DET(params, dt, initial, key = None): # pylint: disable=unused-argument
    """Run one step of the deterministic model"""

    # SSPRK3
    q_1 = initial + dt * _triad_rhs(params, initial)
    q_2 = (3/4) * initial + (1/4) * (q_1 + dt * _triad_rhs(params, q_1))
    y = (1/3) * initial + (2/3) * (q_2 + dt * _triad_rhs(params, q_2))

    return y

################################################################################
################################################################################

def _triad_rhs(params, y):
    ret = params.g * jnp.cross(jnp.conjugate(y), params.D*jnp.conjugate(y))

    return ret

def _triad_rhs_HST(params, y):
    b = jnp.array([params.b_k, params.b_p, params.b_q])
    ret = params.g * jnp.cross(b, params.D*jnp.conjugate(y))

    return ret

def _triad_rhs_EST(params, y):
    b = jnp.array([params.b_k, params.b_p, params.b_q])
    ret = params.g * jnp.cross(jnp.conjugate(y), params.D*b)

    return ret

def _h(k, parity, gamma):
    kappa = k / jnp.linalg.norm(k, ord=2)
    nu = jnp.cross(k, gamma) / jnp.linalg.norm(jnp.cross(k, gamma), ord=2)

    h = jnp.cross(nu, kappa) + 1j * parity  * nu

    return h

def _g(k, p, q, s_k, s_p, s_q, gamma):

    return -0.25*jnp.dot(jnp.cross(_h(p, s_p, gamma), _h(q, s_q, gamma)), _h(k, s_k, gamma))
