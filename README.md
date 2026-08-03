# ambforecast: open reproducible forecasts of ambulance incidents, calls and responses

Documentation: <https://ambmodels.github.io/ambforecast/>

## Setup

Dependencies are pinned in `pyproject.toml`. These can be installed using Python 3.13 and your preferred environment manager.

### Mamba

```bash
mamba create -n ambforecast python=3.13.13
mamba activate ambforecast
pip install -e .
```

### venv

On Windows:

```bash
py -3.13 -m venv .venv
.\venv\Scripts\activate
pip install -e .
```

On Linux or macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Poetry

```bash
poetry env use 3.13
poetry install
poetry shell
```

### uv

```bash
uv venv --python 3.13
source .venv/bin/activate
pip install -e .
```

## Pre-commit

This repository includes a pre-commit hook that checks for the filename of real (private) data, which should never be used here. That analysis belongs in a separate, private repository. If you've accidentally referenced the real data file name in a staged file, the hook will detect it and block the commit, prompting you to remove it before processing.

To activate this hook after cloning this repository, run:

```
pre-commit install
```

## Documentation (local build)

```
great-docs build
great-docs preview
```

## Linting and formatting

This will run on all `.py` files and any `.ipynb` notebooks.

```
ruff format
ruff check --fix
```

## Tests

```
pytest
```

## Citation

See `CITATION.cff`.

## Acknowledgements

This work is part of the [STARS project](https://pythonhealthdatascience.github.io/stars/), supported by the Medical Research Council [grant number MR/Z503915/1] 

This repository builds on the work reported in:

> Monks, T., Harper, A., Allen, M. et al. Forecasting the daily demand for emergency medical ambulances in England and Wales: a benchmark model and external validation. BMC Med Inform Decis Mak 23, 117 (2023). https://doi.org/10.1186/s12911-023-02218-z.

The GitHub repositories from that publication are <https://github.com/TomMonks/swast-benchmarking> and <https://github.com/TomMonks/swast-forecast-tool>.