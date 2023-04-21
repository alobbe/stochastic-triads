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
from itertools import product
import toml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import xarray as xr
import xskillscore as xs

from stochastic_triads.models import get_triad_params
from stochastic_triads.ensemble import Ensemble
from stochastic_triads.filtering import run_forecast_triad


config_path = os.path.realpath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..', 
            'config',
            'forecast',
        )
    )

all_configs = [
    #"all.toml",
    "best.toml",
]

def main(config_filepath, out_dir):
    print(f"Running {os.path.basename(__file__)} with Config: {config_filepath}")

    # Enable Float64 precision
    jax.config.update("jax_enable_x64", True)

    # Load config
    with open(config_filepath, 'r') as f:
        config = toml.load(f)

    ens_models = config['ens_models']
    signal_model = config['signal_model']
    parameters = config['params']
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

    REPS = 5
    keys = jax.random.split(key, REPS)
    with open(os.path.join(out_dir, 'crpss.csv'), 'w') as f:
            f.write("ens_model,b_k,b_p,b_q,mcrps,std_err_mcrpss\n")

    param_model_combinations = product(parameters['b_ks'],
                                        parameters['b_ps'],
                                        parameters['b_qs'],
                                        ens_models)
    for b_k, b_p, b_q, ens_model in param_model_combinations:
        print(f"Running {ens_model} with b_k={b_k}, b_p={b_p}, b_q={b_q}")
        params = get_triad_params(b_k=b_k, b_p=b_p, b_q=b_q,
                                k=parameters['k'],
                                p=parameters['p'],
                                q=parameters['q'],
                                s_k = parameters['s_k'],
                                s_p=parameters['s_p'],
                                s_q=parameters['s_q'],
                                gamma=parameters['gamma'])

        ens_energies = jnp.zeros((n_ens, n_steps, 3, REPS))
        sig_energies = jnp.zeros((n_steps, 3, REPS))
        obs = jnp.zeros((n_steps, 3, REPS))
        for iter_key, rep in zip(keys, range(REPS)):
            key, init_noise_key = jax.random.split(iter_key)

            y0_ensemble = jnp.repeat(y0_ens[None, :], n_ens, axis=0)
            init_ens_noise = jax.random.normal(init_noise_key, (n_ens, 3)) * (init_std)
            y0_ensemble = y0_ensemble + init_ens_noise

            init_ensemble = Ensemble(positions=y0_ensemble, weights=(1/n_ens)*jnp.ones(n_ens), n_ens=n_ens)

            # Run model
            ensembles, signals, observations = run_forecast_triad(init_ensemble, signal_model, init_sig, ens_model, params, dt, n_steps, key, obs_std, filt_interval)

            # collect values
            ens_energies = ens_energies.at[..., rep].set((ensembles * jnp.conjugate(ensembles)).real)
            sig_energies = sig_energies.at[..., rep].set((signals * jnp.conjugate(signals)).real)
            obs = obs.at[..., rep].set(observations)

        forecasts = xr.DataArray(
            ens_energies,
            coords = [jnp.arange(n_ens), jnp.arange(n_steps), jnp.arange(3), jnp.arange(REPS)],
            dims = ["member", "steps", "coord", "reps"],
            name = "Filter"
        )

        all_obs = xr.DataArray(
            sig_energies,
            coords = [jnp.arange(n_steps), jnp.arange(3), jnp.arange(REPS)],
            dims = ["steps", "coord", "reps"],
            name = "Filter"
        )

        crps = xs.crps_ensemble(all_obs, forecasts, dim=("steps", "coord"), member_dim="member").to_numpy()
        rank_hist = xs.rank_histogram(all_obs, forecasts, dim="steps", member_dim="member").to_numpy().astype(int)

        mean_rh = jnp.mean(rank_hist, axis=1)
        stderr_rh = jnp.std(rank_hist, ddof=1, axis=1) / jnp.sqrt(rank_hist.shape[1])

        mcrps = jnp.mean(crps)
        std_err_mcrpss = jnp.std(crps, ddof=1) / jnp.sqrt(crps.shape[0])

        sns.set_theme(style='whitegrid')
        colors = [list(sns.color_palette('deep'))[i] for i in (3,0,2)]
        fig, axs = plt.subplots(3,1)
        for i in range(3):
            axs[i].set_xticks(jnp.arange(n_ens+1))
            axs[i].bar(x=jnp.arange(mean_rh.shape[1]), height = mean_rh[i,:], facecolor=colors[i], yerr = stderr_rh[i,:], capsize = 4, alpha=0.6)
            axs[i].set_title(['Mode k','Mode p','Mode q'][i])
        
        fig.suptitle(f'{ens_model} Rank Histograms b=[{params.b_k}, {params.b_p}, {params.b_q}] CRPS {mcrps:.4f}' + r'$\pm$' + f'{std_err_mcrpss:.4f}')
        plt.tight_layout()
        plt.savefig(
             os.path.join(out_dir, f"{signal_model}_{ens_model}_forecast_N{n_ens}_every{filt_interval}_b{params.b_k:.3f}_{params.b_p:.3f}_{params.b_q:.3f}_rank_hist.png"),
             dpi=300)
        plt.close()

        with open(os.path.join(out_dir, 'crpss.csv'), 'a') as f:
            f.write(f"{ens_model},{params.b_k:.6f},{params.b_p:.6f},{params.b_q:.6f},{mcrps:.6f},{std_err_mcrpss:.6f}\n")

    crps_data = pl.read_csv(os.path.join(out_dir,  'crpss.csv'))
    sub_data = crps_data.pivot(values='mcrps', index=('b_k','b_p','b_q'), columns='ens_model', aggregate_function=None)
    sub_data = sub_data.with_columns(((pl.col('EST')+pl.col('HST'))/2).alias('Mean')).sort('Mean')
    sub_data.write_csv(os.path.join(out_dir, 'crps_sorted.csv'))


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