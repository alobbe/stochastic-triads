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

import toml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from stochastic_triads.models import run_model, get_triad_params

config_path = os.path.realpath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..', 
            'config',
            'trajectory',
        )
    )

all_configs = [
    "sample_DET.toml",
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
    dt = config['dt']
    n_steps = config['n_steps']
    key = jax.random.PRNGKey(config['key'])

    # Run model
    ts = jnp.linspace(0., n_steps * dt, n_steps+1)
    ys = run_model(model, params, dt, y0, n_steps, key)

    # Get energy and helicity
    energies = (ys * jnp.conjugate(ys)).real
    total_energy = jnp.sum(energies, axis=1)
    helicity = jnp.sum(ys * params.D*jnp.conjugate(ys), axis=1).real

    # Plot
    sns.set_theme(style='whitegrid')
    colors = [list(sns.color_palette('deep'))[i] for i in (3,0,2)]
    labels = ('k', 'p', 'q')
    for i in range(3):
        plt.plot(ts, energies[:,i], color=colors[i], label=labels[i])
    plt.plot(ts, total_energy, color = 'black', linestyle='dashed', label='energy')
    plt.plot(ts, helicity, color='black', linestyle='dashdot', label='helicity')
    plt.legend()
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.xlabel('t')
    plt.title(f"{model} Triad Realisation")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{model}_realisation_T{ts[-1]:.0f}.png"),
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