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
        array([[1., 0., 0., 0., 0., 0.],
               [2., 1., 0., 0., 0., 0.],
               [3., 2., 1., 0., 0., 0.]])
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
    Takes an array with dimensions m by k,
    where m represents channels and k represents observations,
    and "extends" it to return an array of shape m by R+1 by k.
    
    Parameters
    ----------
        x_mat: numpy.ndarray
            2D array to be extended.
        R: int
            How far to extend x.
        
    Returns
    -------
        numpy.ndarray
            m x (R+1) x k extended array.
        
    Example:
        >>> R = 5
        >>> x_mat = np.array([[1, 2, 3, 4,], [5, 6, 7, 8,]])
        >>> extend_input_all_channels(x_mat, 3)
        array([[[1., 2., 3., 4.],
                [0., 1., 2., 3.],
                [0., 0., 1., 2.],
                [0., 0., 0., 1.]],

               [[5., 6., 7., 8.],
                [0., 5., 6., 7.],
                [0., 0., 5., 6.],
                [0., 0., 0., 5.]]])

    """
    extended_x_mat = np.zeros([x_mat.shape[0], R + 1, x_mat.shape[1]])

    for i, channel in enumerate(x_mat):
        # Extend channel
        extended_channel = extend_input_by_R(channel, R)

        # Add extended channel to the overall matrix of extended channels
        extended_x_mat[i] = extended_channel

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
    >>> x = np.array([[2,10,2,2],
                      [5,5,5,5],
                      [3,3,3,3]])
    >>> center_matrix(x)
    >>> [[-2.  6. -2. -2.]
         [ 0.  0.  0.  0.]
         [ 0.  0.  0.  0.]]

    """
    x_cent = x - np.mean(x.T, axis=0)
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
        >>> x = np.array([[5,6,7],  # Feature-1
              [23,29,31]]) # Feature-2
        >>> whiten(x)

    Because this function takes such a long time and is the bottleneck of the process,
    it should be considered for optimization.
    One place to start: 
    https://towardsdatascience.com/only-numpy-back-propagating-through-zca-whitening-in-numpy-manual-back-propagation-21eaa4a255fb
    """

    # Subtract Average to make it so that the data is centered around x=0
    x_cent = center_matrix(x)

    # Calculate covariance matrix
    cov_mat = np.cov(x_cent, rowvar=True, bias=True)

    # Eigenvectors
    w, v = linalg.eig(cov_mat)  # values and vectors

    diagw = np.diag(1 / (w ** 0.5))  # or np.diag(1/((w+.1e-5)**0.5))
    diagw = diagw.real.round(4)

    wzca = np.dot(np.dot(np.dot(v, diagw), v.T), x_cent)

    return wzca
