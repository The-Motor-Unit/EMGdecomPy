import numpy as np


def skew(x, der=False):
    """
    Applies contrast function (if der=False) or
    first derivative of contrast function (if der=True)
    to w.
    skew = x^3 / 3

    Parameters
    ----------
        x: float
            Number to apply contrast function to.
        der: boolean
            Whether to apply derivative (or base version).

    Returns
    -------
        float
            Float with contrast function applied.

    Examples
    --------
        >>> x = 4
        >>> skew(x, der=True)
        16
    """

    # first derivative of x^3/3 = x^2
    if der == True:
        rtn = x ** 2
    else:
        rtn = (x ** 3) / 3

    return rtn


def log_cosh(x, der=False):
    """
    Applies contrast function (if der=False) or
    first derivative of contrast function (if der=True)
    to w.
    function = log(cosh(x))
    Parameters
    ----------
        x: float
            Number to apply contrast function to.
        der: boolean
            Whether to apply derivative (or base version).
    Returns
    -------
        float
            Float with contrast function applied.
    Examples
    --------
        >>> x = 4
        >>> log_cosh(x)
        3.3071882258129506
    """

    # first derivative of log(cosh(x)) = tanh(x)
    if der == True:
        rtn = np.tanh(x)
    else:
        x = abs(x)
        if x > 710:  # cosh(x) breaks for abs(x) > 710
            rtn = x - 0.7
        else:
            rtn = np.log(np.cosh(x))

    return rtn


def exp_sq(x, der=False):
    """
    Applies contrast function (if der=False) or
    first derivative of contrast function (if der=True)
    to w.
    function = exp((-x^2/2))

    Parameters
    ----------
        x: float
            Number to apply contrast function to.
        der: boolean
            Whether to apply derivative (or base version).

    Returns
    -------
        float
            Float with contrast function applied.

    Examples
    --------
        >>> x = 4
        >>> exp_sq(4, der=True)
        -0.0013418505116100474
    """

    # first derivative of exp((-x^2/2)) = -e^(-x^2/2) x
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
        >>> apply_contrast(w, fun)
        array([1, 4, 9])
    """

    rtn = fun(w, der)
    return rtn
