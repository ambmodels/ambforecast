# Changelog

All notable changes to this project are documented.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Dates formatted as YYYY-MM-DD as per [ISO standard](https://www.iso.org/iso-8601-date-and-time-format.html).

## v0.2.0 - 2026-09-15

Major refactor, add lots of new functions, and add documentation. Can now run time-series cross-validation with ARIMA, Prophet, Naive, ensemble and ETS models, and then calculate forecast errors and generate various plots. ARIMA and Prophet both support multiple exogenous regressors.

### Added

* Add functions for creating ensemble models (`ensemble.py`), calculating forecast errors (`errors.py`), running ETS models (`ets.py`), naive models (`naive.py`), more plots (`plots.py`), and dividing data for validation (`splits.py`).
* Add `great-docs` documentation site (including GitHub actions to render and deploy docs and lint alt text, plus two pages in `user_guide/`).
* Add `all-contributors`.

### Changed

* **Major refactor!** To improve usability. Changes throughout existing codebase (e.g., `arima.py`, `prophet.py`, `forecast.py`, `processing.py`).
* Improved README (e.g., illustration, further descriptions).
* Dependencies no longer pinned.

## v0.1.0 - 2026-08-05

🌱 First release. Contains refactored code which generates the same ARIMA and Prophet forecasts as provided by SWAST (same parameters, similar plots).