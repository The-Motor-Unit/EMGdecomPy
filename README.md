# EMGdecomPy

[![ci-cd](https://github.com/UBC-SPL-MDS/emgdecompy/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/UBC-SPL-MDS/emgdecompy/actions/workflows/ci-cd.yml)
[![Documentation Status](https://readthedocs.org/projects/emgdecompy/badge/?version=latest)](https://emgdecompy.readthedocs.io/en/stable/?badge=latest)
[![codecov](https://codecov.io/gh/UBC-SPL-MDS/emgdecompy/branch/main/graph/badge.svg?token=78ZU40UEOE)](https://codecov.io/gh/UBC-SPL-MDS/emgdecompy)

A package for decomposing multi-channel intramuscular and surface EMG signals into individual motor unit activity based off the blind source algorithm described in [`Negro et al. (2016)`](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta).

## Table of Contents

- [Overview](#overview)
- [Project Directory](#project-directory)
- [Proposal and Final Report](#proposal-and-final-report)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

## Overview

### What's Been Accomplished

An open-source Python package, `EMGdecomPy` containing two elements, a blind source separation algorithm based on the work of [`Negro et al. (2016)`](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta) and a visualization element, has been created to decompose raw EMG signals into its constituent motor unit activity. Experimental durations of any length can be run using `EMGdecomPy`.

The blind source separation algorithm has been modified slightly from [`Negro et al. (2016)`](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta). The initialization process of the separation vectors has been changed so that instead of initializing every separation vector with the same time instance of highest activity in the pre-processed data, each subsequent vector is initialized with the next highest activity time instance in the pre-processed data.

More customization of the decomposition process is also allowed through different arguments to the `decomposition` function. For example, the separation vectors can be orthogonalized against each other using either the 'source deflation' process described in [`Negro et al. (2016)`](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta) or the Gram-Schmidt method.

We have not had the chance to thoroughly validate our algorithm but preliminary results look promising, as 3 out of 5 of the MUAP shapes identified by `EMGdecomPy` were also identified by [`Hug et al. (2021)`](https://figshare.com/articles/dataset/Analysis_of_motor_unit_spike_trains_estimated_from_high-density_surface_electromyography_is_highly_reliable_across_operators/13695937) for the **Gastrocnemius lateralis** muscle with 10% contraction intensity.  Refer to the [documentation](https://emgdecompy.readthedocs.io/en/stable/autoapi/emgdecompy/decomposition/index.html#emgdecompy.decomposition.decomposition) and the [final report](https://github.com/UBC-SPL-MDS/emg-decomPy/blob/main/docs/final-report/final-report.pdf) for more information.

The visualization element allows the user to interactively visualize the results of the blind source separation algorithm. The user can visualize one motor unit at a time from the motor units that were extracted from the EMG data using the algorithm. The visualization includes four plots, the instantaneous firing rate vs time, the signal vs time, an overlayed version of both the previous plots, and the average motor unit action potential shapes per channel. For a better idea of the interactivity of the plot, refer to the [`EMGdecomPy` workflow notebook](https://github.com/The-Motor-Unit/EMGdecomPy/blob/main/notebooks/emgdecompy-worfklow.ipynb).

### What's Not Working

Currently, the blind source separation algorithm accepts multiple motor units of the same shape. Upon inspection, it can be seen that many of these motor units have the exact same firing times or are time lagged from each other. Solutions to these problems are still in development, and include adding an orthogonalization step to the `refinement` function to stop the refinement process from converging on previous motor units and not accepting motor units whose firing times are within a certain time frame as another motor unit.

There is also a bug in the visualization component that does not allow the user to visualize the results of a decomposition if only one motor unit is accepted. This bug is due to how the peak shapes are created and a fix is currently in development.

### Future Work

Future work includes fixing the aforementioned problems, increasing code efficiency, improving the accuracy of the algorithm using domain knowledge, and further quantitative/qualitative validation of the results of the algorithm using the data from [`Hug et al. (2021)`](https://figshare.com/articles/dataset/Analysis_of_motor_unit_spike_trains_estimated_from_high-density_surface_electromyography_is_highly_reliable_across_operators/13695937) and other EMG data sources.

A further improvement to the algorithm would be a re-learning feature. The user would run the algorithm on a sample of the data, and then identify inaccurate firing times (false positives) based on physiological limits of motor unit firing rates. Then the algorithm would use this information to no longer make similar mistakes in the rest of the decomposition. Implementing this feature would be quite complex because it is algorithmically unclear how this would be done.

One idea is to somehow change the initialization of the separation vectors so that they no longer identify the false firing times when applied to the pre-processed data. However, since the separation vector changes throughout the LCA and refinement processes, it would be hard to control the effect that this would have on the estimated firing times. Another approach would be to influence the KMeans algorithm so that the threshold for the small peaks cluster includes the false positive peaks, in the hopes that future peaks of similar size are also false positives. The downside to this approach would be that we may increase the number of incorrect identifications of large peaks as small peaks, which are discarded.

An improvement to the visualization related to the above improvement would be the ability to remove peaks with a click of a button. This improvement is already in progress and if the re-learning feature is implemented then these two features can be connected.

## Project Directory

- [.githubworkflows](https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/.github/workflows)
  - Contains file for automated testing and publishing of package.
- [data](https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/data)
  - Contains a "raw" subdirectory with the EMG data corresponding to the **Gastrocnemius lateralis** muscle with 10% contraction intensity EMG data from [`Hug et al. (2021)`](https://figshare.com/articles/dataset/Analysis_of_motor_unit_spike_trains_estimated_from_high-density_surface_electromyography_is_highly_reliable_across_operators/13695937).
  - In the future can contain subdirectories pertaining to results from the blind source separation algorithm.
- [docs](https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/docs)
  - Contains files related to the final report, the proposal, and the `ReadtheDocs` documentation.
- [notebooks](https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/notebooks)
  - Contains a Jupyter notebook with the well-documented reproducible workflow that can be used to apply `EMGdecomPy` on EMG data and/or as a guide on how to use the package.
- [src](https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy)
  - Contains the `.py` scripts containing `EMGdecomPy` source code.
- [tests](https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/tests)
  - Contains the tests for the functions within `src`.

## Proposal and Final Report

To generate the proposal and final report locally, ensure that you have R version 4.1.2 or above installed, as well as the RStudio IDE. Then install the necessary dependencies with the following commands:

```
Rscript -e 'install.packages("rmarkdown", repos="http://cran.us.r-project.org")'
Rscript -e 'install.packages("tinytex", repos="http://cran.us.r-project.org")'
Rscript -e 'tinytex::install_tinytex()'
Rscript -e 'install.packages("bookdown", repos="http://cran.us.r-project.org")'
```

### Proposal

Our project proposal can be found [here](https://github.com/UBC-SPL-MDS/emg-decomPy/blob/main/docs/proposal/proposal.pdf).

To generate the proposal locally, run the following command from the root directory after cloning `EMGdecomPy`:

```
Rscript -e "rmarkdown::render('docs/proposal/proposal.Rmd')"
```

Alternatively, if the above doesn't work, install Docker. While Docker is running, run the following command from the root directory after cloning `EMGdecomPy`:

```
docker run --platform linux/amd64 --rm -v /$(pwd):/home/emgdecompy danfke/pandoc-r-bookdown Rscript -e "rmarkdown::render('home/emgdecompy/docs/proposal/proposal.Rmd')"
```

### Final Report

Our final report can be found [here](https://github.com/UBC-SPL-MDS/emg-decomPy/blob/main/docs/final-report/final-report.pdf).

To generate the final report locally, run the following command from the root directory after cloning `EMGdecomPy`:

```
Rscript -e "rmarkdown::render('docs/final-report/final-report.Rmd')"
```

Alternatively, if the above doesn't work, install Docker. While Docker is running, run the following command from the root directory after cloning `EMGdecomPy`:

```
docker run --platform linux/amd64 --rm -v /$(pwd):/home/emgdecompy danfke/pandoc-r-bookdown Rscript -e "rmarkdown::render('home/emgdecompy/docs/final-report/final-report.Rmd')"
```

## Installation

`EMGdecomPy` is compatible with Python versions 3.9 to 3.11.

```bash
pip install emgdecompy
```

## Usage

After installing emgdecompy, refer to the [`EMGdecomPy` workflow notebook](https://github.com/UBC-SPL-MDS/EMGdecomPy/blob/main/notebooks/emgdecompy-worfklow.ipynb) for an example on how to use the package, from loading in the data to visualizing the decomposition results. Clone and run the notebook locally to view and interact with the visualization.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`EMGdecomPy` was created by [Daniel King](github.com/danfke), [Jasmine Ortega](github.com/jasmineortega), [Rada Rudyak](github.com/Radascript), and [Rowan Sivanandam](github.com/Rowansiv). It is licensed under the terms of the [GPLv3 license](https://choosealicense.com/licenses/gpl-3.0/).

## Credits

`EMGdecomPy` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

The blind source separation algorithm in this package was based off of [`Negro et al. (2016)`](https://iopscience.iop.org/article/10.1088/1741-2560/13/2/026027/meta).

The data used for validation was obtained from [`Hug et al. (2021)`](https://figshare.com/articles/dataset/Analysis_of_motor_unit_spike_trains_estimated_from_high-density_surface_electromyography_is_highly_reliable_across_operators/13695937).

[Guilherme Ricioli](https://github.com/guilhermerc) was consulted for his work on [`semg-decomposition`](https://github.com/guilhermerc/semg-decomposition).
