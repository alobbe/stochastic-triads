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
