import numpy as np
from contrast import skew, apply_contrast


def initialize_w(x_ext):
    """
    Initialize new source.
    "For each new source to be estimated,
    the time instant corresponding to the maximum of the squared
    summation of all whitened extended observation vector was
    located and then the projection vector was initialized to the
    whitened [(non extended?)] observation vector at the same time instant."

    Parameters
    ----------
        x_ext: numpy.ndarray
            The whitened extended observation vector.

    Returns
    -------
        numpy.ndarray
            Initialized observation array.

    Examples
    --------
    >>> x_ext = np.array([[1, 2, 3, 4,], [5, 6, 7, 8,]])
    >>> initialize_w(x_ext)
    array([[1., 2., 3., 4.]]])
    """
    return 0

def orthogonalize(w, B):
    """
    Step 2b from Negro et al. (2016): wi(n) = wi(n) - BB{t}*wi(n)
    Note: this is not true orthogonalization, such as the Gram–Schmidt process.
    This is dubbed in paper "source deflation procedure."

    Parameters
    ----------
        w: numpy.ndarray
            Vectors for which we seek orthogonal matrix.
        B: numpy.ndarray
            Matrix to 'deflate' w by.

    Returns
    -------
        numpy.ndarray
            'Deflated' array.

    Examples
    --------
        >>> w = np.array([[5, 6], [23, 29]])
        >>> B = np.array([[3, 3], [3, 3]])
        >>> orthogonalize(w, B)
        array([[-499, -624],
               [-481, -601]])
    """
    w = w - np.dot(B, np.dot(B.T, w))
    return w


def normalize(w):
    """
    Step 2c from Negro et al. (2016): wi(n) = wi(n)/||wi(n)||

    To normalize a matrix means to scale the values
    such that that the range of the row or column values is between 0 and 1.

    Reference : https://www.delftstack.com/howto/numpy/python-numpy-normalize-matrix/

    Parameters
    ----------
        w: numpy.ndarray
            vectors to normalize

    Returns
    -------
        numpy.ndarray
            'normalized' array

    Examples
    --------
        >>> w = np.array([[5, 6], [23, 29]])
        >>> normalize(w)
        array([[0.13217526, 0.15861032],
               [0.60800622, 0.76661653]])

    """
    norms = np.linalg.norm(w)
    w = w / norms
    return w

def separation(z, B, Tolx=10e-4, fun=skew, max_iter=10):
    """
    Fixed point algorithm described in Negro et al. (2016).
    Finds the separation vector for the i-th source.

    Parameters
    ----------
        z: numpy.ndarray
            Product of whitened matrix W obtained in whiten() step and extended.
        B: numpy.ndarray
            Current separation matrix.
        Tolx: numpy.ndarray
            Tolx for element-wise comparison.
        fun: function
            Contrast function to use.
            skew, log_cosh or exp_sq
        max_iter: int > 0
            Maximum iterations for fixed point algorithm.
            When to stop if it doesn't converge.

    Returns
    -------
        numpy.ndarray
            'Deflated' array.

    Examples
    --------
    >>> w_i = separation(z, fun=exp_sq) # where z in extended, whitened, centered emg data

    """
    n = 0
    w_curr = np.random.rand(z.shape[0])
    w_prev = np.random.rand(z.shape[0])

    while np.linalg.norm(np.dot(w_curr.T, w_prev) - 1) > Tolx and n < max_iter:
        w_prev = w_curr

        # -------------------------
        # 2a: Fixed point algorithm
        # -------------------------

        # Calculate A
        # A = average of (der of contrast functio(n transposed prev(w) x z))
        # A = E{g'[w_prev{T}.z]}
        A = np.dot(w_prev.T, z)
        A = apply_contrast(A, fun, True).mean()

        # Calculate new w_curr
        w_curr = np.dot(w_prev.T, z)
        w_curr = apply_contrast(w_curr, fun, False)
        w_curr = np.dot(z, w_curr).mean()
        w_curr = w_curr - A * w_prev

        # -------------------------
        # 2b: Orthogonalize
        # -------------------------
        w_curr = orthogonalize(w_curr, B)

        # -------------------------
        # 2c: Normalize
        # -------------------------
        w_curr = normalize(w_curr)

        # -------------------------
        # 2d: Iterate
        # -------------------------
        n = n + 1

    return w_curr