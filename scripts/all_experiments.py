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
import importlib.util
from argparse import ArgumentParser

_here = os.path.dirname(os.path.realpath(__file__))

all_scripts = [
    "trajectory_sample",
    "trajectory_ensemble",
    "stats_moments",
    "stats_mean",
    "filter",
    "filter_error_stats",
    "forecast",
]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--out_dir',
                        type=str,
                        help='Path to output directory',
                        default = os.path.join(os.path.expanduser("~"), "stochastic-triads-data"),
                        required=False)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    for script_file in all_scripts:
        spec = importlib.util.spec_from_file_location(script_file, os.path.join(_here, f'{script_file}.py'))
        script = importlib.util.module_from_spec(spec) # type: ignore
        spec.loader.exec_module(script) # type: ignore

        for config in script.all_configs:
            script.main(
                os.path.join(script.config_path, config),
                args.out_dir,
            )
