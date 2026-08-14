# ambforecast: open reproducible forecasts of ambulance incidents, calls and responses
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

![](assets/images/forecast.png)

Documentation: <https://ambmodels.github.io/ambforecast/>

## Setup

Dependencies are listed in `pyproject.toml`. These can be installed using Python 3.13 and your preferred environment manager.

### Mamba

```bash
mamba create -n ambforecast python=3.13
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

## Contributors ✨

TODO: Add Lee and Irene once receive GitHub usernames.

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/amyheather"><img src="https://avatars.githubusercontent.com/u/92166537?v=4?s=100" width="100px;" alt="Amy Heather"/><br /><sub><b>Amy Heather</b></sub></a><br /><a href="https://github.com/ambmodels/ambforecast/commits?author=amyheather" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://experts.exeter.ac.uk/19244-thomas-monks"><img src="https://avatars.githubusercontent.com/u/881493?v=4?s=100" width="100px;" alt="Tom Monks"/><br /><sub><b>Tom Monks</b></sub></a><br /><a href="https://github.com/ambmodels/ambforecast/commits?author=TomMonks" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/robchallen"><img src="https://avatars.githubusercontent.com/u/16591648?v=4?s=100" width="100px;" alt="Rob Challen"/><br /><sub><b>Rob Challen</b></sub></a><br /><a href="#ideas-robchallen" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Acknowledgements

This work is part of the [STARS project](https://pythonhealthdatascience.github.io/stars/), supported by the Medical Research Council [grant number MR/Z503915/1] 

This repository builds on the work reported in:

> Monks, T., Harper, A., Allen, M. et al. Forecasting the daily demand for emergency medical ambulances in England and Wales: a benchmark model and external validation. BMC Med Inform Decis Mak 23, 117 (2023). https://doi.org/10.1186/s12911-023-02218-z.

The GitHub repositories from that publication are <https://github.com/TomMonks/swast-benchmarking> and <https://github.com/TomMonks/swast-forecast-tool>.

<br>

```
  ▓▓▓▓▓░░░░░░░░░░░░░▒▒▒▒▒▒░░░░░░░░░░░▒▓▓▓▒░▒▓▓▓▓▒
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓▓▓▓▒░░
  ░░░░░░░░░░░▒██████████████░░░░░░░░░░░░░░░▒▓▓▓▓▒░░░░░░░
  ░░░░░░░░░░░▒██████████████░░░░░░▒▓▓░░░░░░▒▓▓▒░██████████
  ░░░░░░░░░░░▒██████████████░░░░▒▓▒░░▒▓░░░░▒▓▓▒░███▓▓▓▓█████
  ░░░░░░░░░░░▒██████████████░░░░░░▒▓▓░░░░░░▒▓▓▒░███▓▓▓▓██▓▓▓███
  ░░░░░░░░░░░▒██████████████░░░░░░░░░░░░░░░▒▓▓▒░███▓▓▓▓██▓▓▓▓▓███
  ░░░░░░░░░░░▒██████████████░░░░░░░░░░░░░░░▒▓▓▒░███▓▓▓▓██▓▓▓▓▓▓▓███
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓▓▒░██████████████████▓░░
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▓▓▒░░░░░░░░░░░░░░░░░░░░░░░░░
  ▓▓▓▓▓▓▓▓▓▓▓▓▒░░░░░░░░░░▒▓▓▓▓▓▓▓▓▓▓▓░░░░░░▒▓▓░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░▒▒▒
  ▓▓▓▓▓▓▓▓▓▓▓▓▒░░░░░░░░░░▒▓▓▓▓▓▓▓▓▓▓▓░░░░░░▒▓▓░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░▒▒▒
  ░░░░░░░░░░░░▒▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▒▒▒
  ░░░░░░░░░░░░▒▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    ░░░░░░░░░░░░░░▒████▒░░░░░░░░░░░░░░░░░░░▒▓▓░░░░░░░░░░░▓▓▓▓██████▓▓▓▓▓▓▓
             ░░░███▓▓▓▓██▒░░                             ▓▓███▓▓▓▓███▓▓▓▓▓
                ███▓▓▓▓███                                 ███▓▓▓▓███
                  ██████                                     ██████
```