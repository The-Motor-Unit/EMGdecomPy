# Copyright (C) 2022 Daniel King, Jasmine Ortega, Rada Rudyak, Rowan Sivanandam
# This script contains the contrast functions used in the blind source separation algorithm.

import numpy as np
import warnings


def skew(w, der=False):
    """
    Applies contrast function (if der=False) or
    first derivative of contrast function (if der=True)
    to w.
    skew = w^3 / 3

    Parameters
    ----------
        w: np.array
            Array to apply contrast function to.
        der: boolean
            Whether to apply derivative (or base version).

    Returns
    -------
        np.array
            Array with contrast function applied, same shape as w.

    Examples
    --------
        >>> w = np.array([1, 2, 3, 800])
        >>> skew(w, der=True)
        array([1, 4, 9, 640000])
    """

    # first derivitive of x^3/3 = x^2
    if der == True:
        rtn = w ** 2
    else:
        rtn = (w ** 3) / 3

    return rtn


def log_cosh(w, der=False):
    """
    Applies contrast function (if der=False) or
    first derivative of contrast function (if der=True)
    to each element of w.
    function = log(cosh(w))

    Parameters
    ----------
        w: np.array
            Array to apply contrast function to.
        der: boolean
            Whether to apply derivative (or base version).

    Returns
    -------
        np.array
            Array with contrast function applied, same shape as w.

    Examples
    --------
        >>> w = np.array([1, 2, 3, 800])
        >>> log_cosh(w)
        array([4.33780830e-01, 1.32500275e+00, 2.30932850e+00, 7.99300000e+02])
    """

    # First derivitive of log(cosh(x)) = tanh(x)
    if der == True:
        rtn = np.tanh(w)
    else:
        warnings.filterwarnings(
            "ignore"
        )  # To avoid warning from np.cosh(w) for values over 710
        w = abs(w)
        rtn = np.where(w > 710, w - 0.7, np.log(np.cosh(w)))
        warnings.resetwarnings()

    return rtn


def exp_sq(x, der=False):
    """
    Applies contrast function (if der=False) or
    first derivative of contrast function (if der=True)
    to w.
    exp_sq = exp((-x^2/2))

    Parameters
    ----------
        w: np.array
            Array to apply contrast function to.
        der: boolean
            Whether to apply derivative (or base version).

    Returns
    -------
        np.array
            Array with contrast function applied, same shape as w.

    Examples
    --------
        >>> w = np.array([1, 2, 3, 800])
        >>> exp_sq(w, der=False)
        array([0.60653066, 0.13533528, 0.011109, 0.])
    """

    # first derivitive of exp((-x^2/2)) = -e^(-x^2/2) x
    pwr_x = -(x ** 2) / 2
    if der == True:
        rtn = -(np.exp(pwr_x) * x)
    else:
        rtn = np.exp(pwr_x)

    return rtn


def apply_contrast(w, fun=skew, der=False):
    """
    Takes first derivitive and applies contrast function to w
    for Step 2a of fixed point algorithm.
    Options include functions mentioned in Negro et al. (2016).

    Parameters
    ----------
        fun: str
            Name of contrast function to use.
        w: numpy.ndarray
            Matrix to apply contrast function to.

    Returns
    -------
        numpy.ndarray
            Matrix with contrast function applied.

    Examples
    --------
        >>> w = np.array([1, 2, 3])
        >>> fun = skew
        >>> apply_contrast(w, fun, True)
        array([1, 4, 9])

        >>> w = np.array([0.01, 0.1, 1, 10, 100, 1000])
        >>> fun = log_cosh
        >>> apply_contrast(w, fun)
        array([4.99991667e-05, 4.99168882e-03, 4.33780830e-01, 9.30685282e+00,
        9.93068528e+01, 9.99300000e+02])

    """

    rtn = fun(w, der)
    return rtn
