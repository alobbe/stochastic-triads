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
    "DET.toml",
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

    # Load model
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

    # create initial ensemble with spread
    y0_ens = jnp.repeat(y0_ens[None, :], n_ens, axis=0)
    key, init_key = jax.random.split(key)
    init_ens_noise = jax.random.normal(init_key, (n_ens, 3)) * (init_std)
    y0_ens = y0_ens + init_ens_noise
    init_ensemble = Ensemble(positions=y0_ens, weights=(1/n_ens)*jnp.ones(n_ens), n_ens=n_ens)

    # Run model
    ts = jnp.linspace(0., n_steps*filt_interval * dt, n_steps*filt_interval+1)
    ensemble_paths, signal_path, ess_s, observations = run_filter_triad(init_ensemble, signal_model, init_sig, ens_model, params, dt, n_steps, key, obs_std, filt_interval)

    # Get energy and helicity
    ens_energies = (ensemble_paths * jnp.conjugate(ensemble_paths)).real
    sig_energies = (signal_path * jnp.conjugate(signal_path)).real

    assimilation_times = ts[filt_interval::filt_interval]

    # Plot
    sns.set_theme(style='whitegrid')
    colors = [list(sns.color_palette('deep'))[i] for i in (3,0,2)]
    modes = ('k', 'p', 'q')
    for i in range(3):
        for p in range(n_ens):
            plt.plot(ts, ens_energies[p, :, i], color=colors[i], alpha=0.4, linewidth=0.6)
        plt.plot(ts, sig_energies[:, i], color='gray', linewidth=1.5)
        plt.plot(assimilation_times, observations[:,i], '*', color='black')
        plt.xlabel('t')
        plt.ylim([-0.025,0.85])
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.25))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(35))
        plt.title(f'Particle Filter Signal {signal_model} Ensemble {ens_model} N={n_ens} Mode {modes[i]}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{signal_model}_{ens_model}_filtering_N{n_ens}_every{filt_interval}_mode{modes[i]}.png"), 
            dpi=300)
        plt.close()

    # Plot ESS
    with sns.axes_style('white'):
        plt.bar(assimilation_times, ess_s,
                color=sns.color_palette('deep')[1],
                width = 2)
        plt.xlabel('t')
        plt.ylabel('ESS')
        plt.title(f'ESS Particle Filter Signal {signal_model} Ensemble {ens_model} N={n_ens}')
        plt.tight_layout()
        sns.despine()
        plt.savefig(
            os.path.join(out_dir, f"{signal_model}_{ens_model}_filtering_N{n_ens}_every{filt_interval}_ESS.png"),
            dpi=300)
        plt.close()

    bias = jnp.abs(jnp.mean(ens_energies, axis=0) - sig_energies)
    rmse = jnp.sqrt(jnp.mean((ens_energies - sig_energies[None,:,:])**2, axis=0))

    for i in range(3):
        plt.plot(ts, bias[:,i], color=colors[i])
        plt.xlabel('t')
        plt.ylim([0.0,0.4])
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(35))
        plt.title(f'Bias Particle Filter Signal {signal_model} Ensemble {ens_model} N={n_ens} Mode {modes[i]}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{signal_model}_{ens_model}_filtering_N{n_ens}_every{filt_interval}_mode{modes[i]}_BIAS.png"),
            dpi=300)
        plt.close()

    for i in range(3):
        plt.plot(ts, rmse[:,i], color=colors[i])
        plt.xlabel('t')
        plt.ylim([0.0,0.4])
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(35))
        plt.title(f'RMSE Particle Filter Signal {signal_model} Ensemble {ens_model} N={n_ens} Mode {modes[i]}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir,f"{signal_model}_{ens_model}_filtering_N{n_ens}_every{filt_interval}_mode{modes[i]}_RMSE.png"),
            dpi=300)
        plt.close()

    if ens_model == "DET":
        n_unique = jnp.zeros(n_steps+1)
        for i in track(range(n_steps+1), description="Computing unique ensemble members"):
            _, uniques = jnp.unique(ensemble_paths[:,i*filt_interval,:], axis=0, return_counts=True)
            n_unique = n_unique.at[i].set(len(uniques))
        with sns.axes_style('white'):
            plt.bar(ts[::filt_interval], n_unique,
                    color=sns.color_palette('deep')[1],
                    width = 2)
            plt.xlabel('t')
            plt.title('Number of unique ensemble members')
            plt.tight_layout()
            #plt.gca().set_axisbelow(True)
            sns.despine()
            plt.savefig(
                os.path.join(out_dir, f"{signal_model}_{ens_model}_filtering_N{n_ens}_every{filt_interval}_unique.png"),
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