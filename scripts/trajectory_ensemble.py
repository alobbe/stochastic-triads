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

import os
from argparse import ArgumentParser
from itertools import product

import toml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from stochastic_triads.models import get_triad_params
from stochastic_triads.ensemble import run_ensemble

config_path = os.path.realpath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..', 
            'config',
            'trajectory',
            'ensemble',
        )
    )

all_configs = [
    "fullnoise_EST.toml",
    "k_EST.toml",
    "p_EST.toml",
    "q_EST.toml",
    "fullnoise_HST.toml",
    "k_HST.toml",
    "p_HST.toml",
    "q_HST.toml",
]

def main(config_filepath, out_dir):
    print(f"Running {os.path.basename(__file__)} with Config: {config_filepath}")

    # Enable Float64 precision
    jax.config.update("jax_enable_x64", True)

    # Load config
    with open(config_filepath, 'r') as f:
        config = toml.load(f)

    model = config['model']
    params = get_triad_params(**(config['params']))
    y0 = jnp.array(config['initial_conditions'], dtype=jnp.complex_)
    n_ens = config['n_ens']
    dt = config['dt']
    n_steps = config['n_steps']
    key = jax.random.PRNGKey(config['key'])

    # Run model
    y0 = jnp.repeat(y0[None, :], n_ens, axis=0)
    ts = jnp.linspace(0., n_steps * dt, n_steps+1)
    ens = run_ensemble(model, params, dt, y0, n_steps, key)

    # Get energy and helicity
    energies = (ens * jnp.conjugate(ens)).real
    total_energy = jnp.sum(energies, axis=2)
    helicity = jnp.sum(ens * params.D[None, None, :]*jnp.conjugate(ens), axis=2).real

    # Plot
    sns.set_theme(style='whitegrid')
    colors = [list(sns.color_palette('deep'))[i] for i in (3,0,2)]
    for i,p in product(range(3), range(n_ens)):
        plt.plot(ts, energies[p,:,i], color=colors[i], alpha=0.4, linewidth=0.6)
    for p in range(n_ens):
        plt.plot(ts, total_energy[p,:], color = 'black', alpha=0.4, linewidth=0.6)
        plt.plot(ts, helicity[p,:], color='gray', alpha=0.4, linewidth=0.6)
    for i in range(3):
        plt.plot(ts, jnp.mean(energies[:, :, i], axis=0), color=colors[i],  linewidth=1.5)
    plt.plot(ts, jnp.mean(total_energy, axis=0), color="black",  linewidth=1.5)
    plt.plot(ts, jnp.mean(helicity, axis=0), color="grey",  linewidth=1.5)
    plt.xlabel('t')
    plt.ylim([-1.5, 1.5])
    plt.title(f"{model} Triad Ensemble b=[{params.b_k}, {params.b_p}, {params.b_q}] N={n_ens}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{model}_ensemble_b{params.b_k:.2f}_{params.b_p:.2f}_{params.b_q:.2f}_N{n_ens}.png"),
        dpi=300)

    plt.close()

if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='Path to config file')
    parser.add_argument('--out_dir',
                        type=str,
                        help='Path to output directory',
                        default = os.path.join(os.path.expanduser("~"), "stochastic-triads-data"),
                        required=False)
    args = parser.parse_args()

    # Create out_dir
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Run
    if args.config == "all":
        for config in all_configs:
            main(os.path.join(config_path, config), args.out_dir)
    else:
        main(args.config, args.out_dir)
