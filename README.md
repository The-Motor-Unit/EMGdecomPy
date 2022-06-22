# EMGdecomPy

[![ci-cd](https://github.com/UBC-SPL-MDS/emgdecompy/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/UBC-SPL-MDS/emgdecompy/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/UBC-SPL-MDS/emgdecompy/branch/main/graph/badge.svg?token=78ZU40UEOE)](https://codecov.io/gh/UBC-SPL-MDS/emgdecompy)

A package for decomposing multi-channel intramuscular and surface EMG signals into individual motor unit activity based off the blind source algorithm described in [`Negro et al. (2016)](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta).

## Proposal

Our project proposal can be found [here](https://github.com/UBC-SPL-MDS/emg-decomPy/blob/main/docs/proposal/proposal.pdf).

To generate the proposal locally, run the following command from the root directory:

```Rscript -e "rmarkdown::render('docs/proposal/proposal.Rmd')"```

## Installation

```bash
pip install emgdecompy
```

## Usage and Example

After installing `EMGdecomPy`, to use the package import it with the following commands:

```
from emgdecompy.decomposition import decomposition
```

**Run the blind source separation algorithm on your data to extract out the separation vectors, motor unit firing times, and associated silhouette scores and pulse-to-noise ratios.**

```
decomposition(data)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`EMGdecomPy` was created by [Daniel King](github.com/danfke), [Jasmine Ortega](github.com/jasmineortega), [Rada Rudyak](github.com/Radascript), and [Rowan Sivanandam](github.com/Rowansiv). It is licensed under the terms of the GPLv3 license.

## Credits

`EMGdecomPy` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

The blind source separation algorithm in this package was based off of [`Negro et al. (2016)](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta).

The data used for validation was obtained from [`Hug et al. (2021)`](https://figshare.com/articles/dataset/Analysis_of_motor_unit_spike_trains_estimated_from_high-density_surface_electromyography_is_highly_reliable_across_operators/13695937).

[Guilherme Ricioli](https://github.com/guilhermerc) was consulted for his work on [`semg-decomposition`](https://github.com/guilhermerc/semg-decomposition).
