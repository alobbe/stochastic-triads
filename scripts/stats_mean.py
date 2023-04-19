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

from rich.progress import track
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
            'stats',
            'mean',
        )
    )

all_configs = [
    "large_noise_EST.toml",
    "large_noise_HST.toml",
    "strong_noise_EST.toml",
    "many_particle_EST.toml",
    "many_particle_HST.toml",
]

def main(config_filepath, out_dir):
    print(f"Running {os.path.basename(__file__)} with Config: {config_filepath}")

    # Enable Float64 precision
    jax.config.update("jax_enable_x64", True)

    # Load config
    with open(config_filepath, 'r') as f:
        config = toml.load(f)

    # Load model
    model = config['model']
    params = get_triad_params(**(config['params']))
    y0 = jnp.array(config['initial_conditions'], dtype=jnp.complex_)
    n_ens = config['n_ens']
    dt = config['dt']
    n_steps = config['n_steps']
    key = jax.random.PRNGKey(config['key'])

    # Run model
    ts = jnp.linspace(0., n_steps * dt, n_steps+1)

    PARTICLE_BATCH = 1000
    if n_ens > PARTICLE_BATCH:
        n_batches = n_ens // PARTICLE_BATCH
        keys = jax.random.split(key, n_batches)
        y0 = jnp.repeat(y0[None, :], PARTICLE_BATCH, axis=0)
        mean = jnp.zeros((n_steps+1, 3))
        total_energy_mean = jnp.zeros(n_steps+1)
        helicity_mean = jnp.zeros(n_steps+1)
        for i in track(range(n_batches), description="Running ensemble batches"):
            key = keys[i]
            ens = run_ensemble(model, params, dt, y0, n_steps, key)
            energies = (ens * jnp.conjugate(ens)).real
            mean += jnp.mean(energies, axis=0)
            total_energy_mean += jnp.mean(jnp.sum(energies, axis=2), axis=0)
            helicity_mean += jnp.mean(jnp.sum(ens * params.D[None, None, :]*jnp.conjugate(ens), axis=2).real, axis=0)
        mean /= n_batches
        total_energy_mean /= n_batches
        helicity_mean /= n_batches
    else:
        y0 = jnp.repeat(y0[None, :], n_ens, axis=0)
        ens = run_ensemble(model, params, dt, y0, n_steps, key)

        # Energies
        energies = (ens * jnp.conjugate(ens)).real
        total_energy = jnp.sum(energies, axis=2)
        helicity = jnp.sum(ens * params.D[None, None, :]*jnp.conjugate(ens), axis=2).real

        # Stats
        mean = jnp.mean(energies, axis=0)
        total_energy_mean = jnp.mean(total_energy, axis=0)
        helicity_mean = jnp.mean(helicity, axis=0)

    # Plot
    sns.set_theme(style='whitegrid')
    colors = [list(sns.color_palette('deep'))[i] for i in (3,0,2)]
    for i in range(3):
        plt.plot(ts, mean[:,i], color=colors[i])
    plt.plot(ts, total_energy_mean, color='black')
    plt.plot(ts, helicity_mean, color='grey')
    plt.title(f"{model} Mean b=[{params.b_k}, {params.b_p}, {params.b_q}] N={n_ens}")
    plt.xlabel('t')
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{model}_mean_b{params.b_k:.2f}_{params.b_p:.2f}_{params.b_q:.2f}_N{n_ens}.png"),
        dpi=300)
    plt.close()


if __name__ =="__main__":
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