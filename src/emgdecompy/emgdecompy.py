from scipy.io import loadmat
from scipy import linalg
import pandas as pd
import altair as alt
import numpy as np


def flatten_signal(raw):
    """
    Takes the raw EMG signal array, flattens it, and removes empty channels with no data.

    Parameters
    ----------
    raw: numpy.ndarray
        Raw EMG signal array.

    Returns
    -------
    numpy.ndarray
        Flattened EMG signal array, with empty channels removed.
    """
    # Flatten input array
    raw_flattened = raw.flatten()
    # Remove empty channels and then removes dimension of size 1
    raw_flattened = np.array(
        [channel for channel in raw_flattened if 0 not in channel.shape]
    ).squeeze()

    return raw_flattened


def extend_input_by_R(x, R):
    """
    Takes a one-dimensional array and extends it using past observations.

    Parameters
    ----------
        x: numpy.ndarray
            1D array to be extended.
        R: int
            How far to extend x.
    Returns
    -------
        numpy.ndarray
            len(x) by R+1 extended array.

    Examples
    --------
        >>> R = 5
        >>> x = np.array([1, 2, 3])
        >>> extend_input_by_R(x, R)
        array([[1., 2., 3.],
               [0., 1., 2.],
               [0., 0., 1.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

    """

    # Create array with R+1 rows and length of x + R columns
    extended_x = np.zeros((R + 1, len(x) + R))

    # Create array where each row is a delayed version of the previous row
    for i in range(R + 1):
        extended_x[i][i : i + len(x)] = x

    # Optional: Cut off extra R rows
    extended_x = extended_x.T[0 : len(x)].T

    return extended_x


def extend_all_channels(x_mat, R):
    """
    Takes an array with dimensions M by K,
    where M represents number of channels and K represents observations,
    and "extends" it to return an array of shape M * (R+1) by K.

    Parameters
    ----------
        x_mat: numpy.ndarray
            2D array to be extended.
        R: int
            How far to extend x.

    Returns
    -------
        numpy.ndarray
            M(R+1) x K extended array.

    Examples
    --------
        >>> R = 5
        >>> x_mat = np.array([[1, 2, 3, 4,], [5, 6, 7, 8,]])
        >>> extend_input_all_channels(x_mat, 3)
        array([[1., 2., 3., 4.],
               [0., 1., 2., 3.],
               [0., 0., 1., 2.],
               [0., 0., 0., 1.],
               [5., 6., 7., 8.],
               [0., 5., 6., 7.],
               [0., 0., 5., 6.],
               [0., 0., 0., 5.]])

    """
    extended_x_mat = np.zeros([x_mat.shape[0], (R + 1), x_mat.shape[1]])

    for i, channel in enumerate(x_mat):
        # Extend channel
        extended_channel = extend_input_by_R(channel, R)

        # Add extended channel to the overall matrix of extended channels
        extended_x_mat[i] = extended_channel

    # Reshape to get rid of channels
    extended_x_mat = extended_x_mat.reshape(x_mat.shape[0] * (R + 1), x_mat.shape[1])

    return extended_x_mat


def center_matrix(x):
    """
    Subtract mean of each row.
    Results in the data being centered around x=0.

    Parameters
    ----------
        x: numpy.ndarray
            matrix of arrays to be centered

    Returns
    -------
        numpy.ndarray
            centered matrix array

    Examples
    --------
    >>> x = np.array([[1, 2, 3], [4, 6, 8]])
    >>> center_matrix(x)
    array([[-1.,  0.,  1.],
           [-2.,  0.,  2.]])
    """
    x_cent = x.T - np.mean(x.T, axis=0)
    x_cent = x_cent.T
    return x_cent


def whiten(x):
    """
    Whiten the input matrix.
    First, the data is centred by subtracting the mean and then ZCA whitening is performed.

    Parameters
    ----------
        x: numpy.ndarray
            2D array to be whitened

    Returns
    -------
        numpy.ndarray
            whitened array

    Examples
    --------
        >>> x = np.array([[1, 2, 3, 4],  # Feature-1
                          [5, 6, 7, 8]]) # Feature-2
        >>> whiten(x)
        array([[-0.94874998, -0.31624999,  0.31624999,  0.94874998],
               [-0.94875001, -0.31625   ,  0.31625   ,  0.94875001]])
    """

    # Subtract Average to make it so that the data is centered around x=0
    x_cent = center_matrix(x)

    # Calculate covariance matrix
    cov_mat = np.cov(x_cent, rowvar=True, bias=True)

    # Eigenvalues and eigenvectors
    w, v = linalg.eig(cov_mat)

    # Apply regularization factor, which is the average of smallest half of the eigenvalues (still not sure)
    # w += w[:len(w) / 2].mean()

    # Diagonal matrix inverse square root of eigenvalues
    diagw = np.diag(1 / (w ** 0.5))
    diagw = diagw.real.round(4)

    # Whitening using zero component analysis: v diagw v.T x_cent
    wzca = np.dot(np.dot(np.dot(v, diagw), v.T), x_cent)

    return wzca


def orthogonalize(w, B):
    """
    Step 2b from Negro et al: wi(n) = wi(n) - BB{t}*wi(n)
    Note: this is not true orthogonalization, such as Gram–Schmidt process
    This is dubbed in paper "source deflation procedure"

    Parameters
    ----------
        w: numpy.ndarray
            vectors for which we seek orthogonal matrix
        B: numpy.ndarray
            matrix to 'deflate' w by

    Returns
    -------
        numpy.ndarray
            'deflated' array

    Example
    --------
        >>> w = np.array([[5,6],
              [23,29]])
        >>> B = np.array([[3,3],
              [3,3]])
        >>> orthogonalize(w, B)
    """
    w = w - np.dot(np.dot(B.T, w), B)
    return w


def normalize(w):
    """
    Step 2c from Negro et al: wi(n) = wi(n)/||wi(n)||

    To normalize a matrix means to scale the values
    such that that the range of the row or column values is between 0 and 1

    Reference : https://www.delftstack.com/howto/numpy/python-numpy-normalize-matrix/

    Parameters
    ----------
        w: numpy.ndarray
            vectors to normalize

    Returns
    -------
        numpy.ndarray
            'normalized' array

    Example
    --------
        >>> w = np.array([[5,6],
              [23,29]])
        >>> normalize(w)
    """
    norms = np.linalg.norm(w)
    w = w / norms
    return w


def apply_contrast_fun_router(w, fun=skew):
    """
    Takes first derivitive and applies contrast function to w with map()
    for Step 2a of fixed point algorithm
    Options include functions mentioned in Negro et al

    Parameters
    ----------
        fun: str
            name of contrast function to use
        w: numpy.ndarray
            matrix to apply contrast function to

    Returns
    -------
        numpy.ndarray
            matrix with contrast function applied

    Example
    --------
        >>> w = np.array([1, 2, 3])
        >>> fun = skew
        >>> apply_contrast_fun_router(w, fun)
        >>> array([1, 4, 9])
    """

    # an_array = np.array([1, 2, 3])

    # def double(x):
    #    return x * 2

    # mapped_array = double(an_array)
    # print(mapped_array)

    rtn = fun(w)
    return rtn


def skew(x):
    """
    Takes first derivitive and applies skew function to w
    skew = x^3 / 3

    Parameters
    ----------
        x: numpy.ndarray
            matrix to apply contrast function to

    Returns
    -------
        numpy.ndarray
            matrix with contrast function applied

    Example
    --------
        >>> x = 4
        >>> skew(x)
        >>> 16
    """

    # first derivitive of x^3/3 = x^2
    rtn = x ** 2

    return rtn


def separation(z, Tolx):
    """
    Parameters
    ----------
        z: numpy.ndarray
            Product of whitened matrix W obtained in whiten() step and extended
        Tolx: numpy.ndarray
            Tolx for element-wise comparison

    Returns
    -------
        numpy.ndarray
            'deflated' array
    """
    n = 0
    # while ()

    # calculate A
    # A = average of

    w = fixed_point_algo(w, z)  # 2a
    w = orthogonalize(w, B)  # 2b
    w = normalize(w)  # 2c
    n = n + 1  # 2d
