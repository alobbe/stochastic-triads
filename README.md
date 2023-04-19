# Stochastic Triad Models of Turbulence :trident:

[![DOI](https://zenodo.org/badge/629953029.svg)](https://zenodo.org/badge/latestdoi/629953029)

Simulate stochastic triad models and perform particle filtering, model calibration and forecasting.

Triad models are classical reduced order models of turbulence in fluid dynamics.

To learn more about stochastic triad models and why they are useful in fluid dynamics, please refer to the associated research paper *Comparison of Stochastic Parametrization Schemes using Data Assimilation on Triad Models* by Bertrand Chapron, Dan Crisan, Darryl Holm, Oana Lang, Alexander Lobbe, and Etienne Mémin.

The code uses Google's excellent [JAX](https://github.com/google/jax) library. It is organized around a performant simulator core that is packaged as a python package named *stochastic_triads*. The ensemble simulations computed by the core are accelerated using the [`jax.jit`](https://github.com/google/jax#compilation-with-jit) (just-in-time compilation) and [`jax.vmap`](https://github.com/google/jax#auto-vectorization-with-vmap) (vectorization) functionalities.

## Installation

The code was developed and tested using **Python 3.10.6**. To check your version of python run
```
python3 --version
```

The recommended way to install is within a virtual environment using python version 3.10.6.

1. Create a new virtual environment.
    ```
    python3 -m venv env
    ```

2. Activate the virtual environment.
    - On Mac/Linux:
    ```
    source env/bin/activate
    ```
    - On Windows:
    ```
    env\Scripts\activate.bat
    ```

3. Download and unpack the repo.
    - On Mac/Linux:
    ```
    curl -L https://github.com/alobbe/stochastic-triads/archive/main.tar.gz | tar -xz
    ```
    - On Windows:
    ```
    Invoke-WebRequest -Uri "https://github.com/alobbe/stochastic-triads/archive/main.zip" -OutFile "stochastic-triads.zip"; Expand-Archive "stochastic-triads.zip" -DestinationPath "./stochastic-triads"
    ```

4. Install the core simulator package *stochastic_triads* using `pip`.
    ```
    cd stochastic-triads
    pip install .
    ```

5. Install required packages for running the scripts to perform the experiments.
    ```
    pip install -r requirements.txt
    ```

## Usage

The scripts to run experiments corresponding to the paper are located in the `scripts/` directory.

The `config/` directory contains the example configurations for experiments corresponding to the paper linked above.

To run any of the provided experiment scripts (except `all_experiments.py`) from within the `stochastic-triads/` directory, run
```
python3 scripts/<scriptname>.py <path/to/config_file.toml>
```
where you specify the config file you wish to run. To run all experiments from the paper corresponding to a given script you can simply use `all`:
```
python3 scripts/<scriptname>.py all
```

If you wish to run *all* experiments for *all* example configs use the provided `all_experiments.py` script like so:
```
python3 scripts/all_experiments.py
```

By default, the generated data will be saved in the directory `~/stochastic-triads-data/`, located within your home folder. You can use the optional `--out-dir` flag on any of the above commands to specify an alternative location.

## Citation

If you use this code in your research, please cite it using

```
@article{alobbe_2023, title={alobbe/stochastic-triads: Initial Release}, DOI={10.5281/zenodo.7845270}, publisher={Zenodo}, author={Alexander Lobbe}, year={2023}, month={Apr} }
```