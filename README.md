# EMGdecomPy

[![ci-cd](https://github.com/UBC-SPL-MDS/emgdecompy/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/UBC-SPL-MDS/emgdecompy/actions/workflows/ci-cd.yml)
[![Documentation Status](https://readthedocs.org/projects/emgdecompy/badge/?version=latest)](https://emgdecompy.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/UBC-SPL-MDS/emgdecompy/branch/main/graph/badge.svg?token=78ZU40UEOE)](https://codecov.io/gh/UBC-SPL-MDS/emgdecompy)

A package for decomposing multi-channel intramuscular and surface EMG signals into individual motor unit activity based off the blind source algorithm described in [`Negro et al. (2016)`](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta).

## Proposal and Final Report

To generate the proposal and final report locally, ensure that you have R version 4.1.2 or above installed, as well as the RStudio IDE. Then install the necessary dependencies with the following commands:

```
Rscript -e 'install.packages("rmarkdown")'
Rscript -e 'install.packages("tinytex")'
Rscript -e 'tinytex::install_tinytex()'
Rscript -e 'install.packages("bookdown")'
```

### Proposal

Our project proposal can be found [here](https://github.com/UBC-SPL-MDS/emg-decomPy/blob/main/docs/proposal/proposal.pdf).

To generate the proposal locally, run the following command from the root directory after cloning `EMGdecomPy`:

```Rscript -e "rmarkdown::render('docs/proposal/proposal.Rmd')"```

Alternatively, if the above doesn't work, install Docker. While Docker is running, run the following command from the root directory after cloning `EMGdecomPy`:

```docker run --platform linux/amd64 --rm -v /$(pwd):/home/emgdecompy danfke/pandoc-r-bookdown Rscript -e "rmarkdown::render('home/emgdecompy/docs/proposal/proposal.Rmd')"```

### Final Report

Our final report can be found [here](https://github.com/UBC-SPL-MDS/emg-decomPy/blob/main/docs/final-report/final-report.pdf).

To generate the final report locally, run the following command from the root directory after cloning `EMGdecomPy`:

```Rscript -e "rmarkdown::render('docs/final-report/final-report.Rmd')"```

Alternatively, if the above doesn't work, install Docker. While Docker is running, run the following command from the root directory after cloning `EMGdecomPy`:

```docker run --platform linux/amd64 --rm -v /$(pwd):/home/emgdecompy danfke/pandoc-r-bookdown Rscript -e "rmarkdown::render('home/emgdecompy/docs/final-report/final-report.Rmd')"```

## Installation

`EMGdecomPy` is compatible with Python versions 3.9 to 3.11.

```bash
pip install emgdecompy
```

## Usage

After installing emgdecompy, refer to the [`EMGdecomPy` workflow notebook](https://github.com/UBC-SPL-MDS/EMGdecomPy/blob/main/notebooks/emgdecompy-worfklow.ipynb) for an example on how to use the package, from loading in the data to visualizing the decomposition results.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`EMGdecomPy` was created by [Daniel King](github.com/danfke), [Jasmine Ortega](github.com/jasmineortega), [Rada Rudyak](github.com/Radascript), and [Rowan Sivanandam](github.com/Rowansiv). It is licensed under the terms of the [GPLv3 license](https://choosealicense.com/licenses/gpl-3.0/).

## Credits

`EMGdecomPy` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

The blind source separation algorithm in this package was based off of [`Negro et al. (2016)`](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta).

The data used for validation was obtained from [`Hug et al. (2021)`](https://figshare.com/articles/dataset/Analysis_of_motor_unit_spike_trains_estimated_from_high-density_surface_electromyography_is_highly_reliable_across_operators/13695937).

[Guilherme Ricioli](https://github.com/guilhermerc) was consulted for his work on [`semg-decomposition`](https://github.com/guilhermerc/semg-decomposition).
