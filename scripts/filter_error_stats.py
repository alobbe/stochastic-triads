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

import os
from argparse import ArgumentParser
from rich.progress import track
import toml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from stochastic_triads.ensemble import Ensemble
from stochastic_triads.models import get_triad_params
from stochastic_triads.filtering import run_filter_triad

config_path = os.path.realpath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..', 
            'config',
            'filter',
        )
    )

all_configs = [
    "EST.toml",
    "HST.toml",
    "many_particles_EST.toml",
    "small_interval_EST.toml",
]

def main(config_filepath, out_dir):
    print(f"Running {os.path.basename(__file__)} with Config: {config_filepath}")

    # Enable Float64 precision
    jax.config.update("jax_enable_x64", True)

    # Load config
    with open(config_filepath, 'r') as f:
        config = toml.load(f)

    ens_model = config['ens_model']
    signal_model = config['signal_model']
    params = get_triad_params(**(config['params']))
    n_ens = config['n_ens']
    n_steps = config['n_steps']
    filt_interval = config['filt_interval']
    obs_std = jnp.array(config['obs_std'])
    y0_ens = jnp.array(config['initial_ens'], dtype=jnp.complex_)
    init_std = config['initial_ens_std']
    init_sig = jnp.array(config['initial_signal'], dtype=jnp.complex_)
    dt = config['dt']
    n_steps = config['n_steps']
    key = jax.random.PRNGKey(config['key'])

    REPS = 10
    keys = jax.random.split(key, REPS)

    biases = jnp.zeros((REPS, filt_interval*n_steps+1, 3))
    rmses =  jnp.zeros((REPS, filt_interval*n_steps+1, 3))
    ensembles = jnp.zeros((REPS, n_ens, filt_interval*n_steps+1,  3))
    for rep in track(range(REPS), description="Running reps"):
        key, init_noise_key = jax.random.split(keys[rep])

        y0_ensemble = jnp.repeat(y0_ens[None, :], n_ens, axis=0)
        init_ens_noise = jax.random.normal(init_noise_key, (n_ens, 3)) * (init_std)
        y0_ensemble = y0_ensemble + init_ens_noise

        init_ensemble = Ensemble(positions=y0_ensemble, weights=(1/n_ens)*jnp.ones(n_ens), n_ens=n_ens)

        # Run model
        ensemble_paths, signal_path, _, _ = run_filter_triad(init_ensemble, signal_model, init_sig, ens_model, params, dt, n_steps, key, obs_std, filt_interval)

        # Get energy and helicity
        ens_energies = (ensemble_paths * jnp.conjugate(ensemble_paths)).real
        sig_energies = (signal_path * jnp.conjugate(signal_path)).real

        bias = jnp.abs(jnp.mean(ens_energies, axis=0) - sig_energies)
        rmse = jnp.sqrt(jnp.mean((ens_energies - sig_energies[None,:,:])**2, axis=0))

        biases = biases.at[rep, :, :].set(bias)
        rmses = rmses.at[rep, :, :].set(rmse)
        ensembles = ensembles.at[rep, ...].set(ens_energies)

    ts = jnp.linspace(0., n_steps*filt_interval * dt, n_steps*filt_interval+1)

    mean_bias = jnp.mean(biases, axis=0)
    top_bias = jnp.max(biases, axis=0) # Get the maximum bias for each time step
    bottom_bias = jnp.min(biases, axis=0) # Get the minimum bias for each time step

    mean_rmse = jnp.mean(rmses, axis=0)
    top_rmse = jnp.max(rmses, axis=0) # Get the maximum bias for each time step
    bottom_rmse = jnp.min(rmses, axis=0) # Get the minimum bias for each time step

    mean_ensemble = jnp.mean(ensembles, axis=1) # Average over ensemble members
    mean_mean_ensemble = jnp.mean(mean_ensemble, axis=0) # Average over reps

    # Plot
    sns.set_theme(style='whitegrid')
    colors = [list(sns.color_palette('deep'))[i] for i in (3,0,2)]
    particle_colors = [list(sns.color_palette('pastel'))[i] for i in (3,0,2)]
    modes = ('k', 'p', 'q')

    for i in range(3):
        for j in range(REPS):
            plt.plot(ts, mean_ensemble[j,:,i], color=particle_colors[i], linewidth=0.7)
        plt.plot(ts, mean_mean_ensemble[:,i], color=colors[i])
        plt.plot(ts, sig_energies[:,i], color='grey', linewidth=1.2, linestyle='--')
        plt.xlabel('t')
        plt.ylim([0.0,0.8])
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(35))
        plt.title(f'Mean Particle Filter Signal {signal_model} Ensemble {ens_model} N={n_ens} Mode {modes[i]}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{signal_model}_{ens_model}_filtering_N{n_ens}_every{filt_interval}_mode{modes[i]}_MEAN_stat.png"),
            dpi=300)
        plt.close()

    for i in range(3):
        plt.plot(ts, mean_bias[:,i], color=colors[i])
        plt.fill_between(ts, top_bias[:,i], bottom_bias[:,i], color=particle_colors[i])
        plt.xlabel('t')
        plt.ylim([0.0,0.4])
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(35))
        plt.title(f'Bias Particle Filter Signal {signal_model} Ensemble {ens_model} N={n_ens} Mode {modes[i]}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{signal_model}_{ens_model}_filtering_N{n_ens}_every{filt_interval}_mode{modes[i]}_BIAS_stat.png"),
            dpi=300)
        plt.close()

    for i in range(3):
        plt.plot(ts, mean_rmse[:,i], color=colors[i])
        plt.fill_between(ts, top_rmse[:,i], bottom_rmse[:,i] , color=particle_colors[i])
        plt.xlabel('t')
        plt.ylim([0.0,0.4])
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(35))
        plt.title(f'RMSE Particle Filter Signal {signal_model} Ensemble {ens_model} N={n_ens} Mode {modes[i]}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{signal_model}_{ens_model}_filtering_N{n_ens}_every{filt_interval}_mode{modes[i]}_RMSE_stat.png"),
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