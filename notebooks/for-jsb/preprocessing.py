from scipy import linalg
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
        >>> R = 3
        >>> x_mat = np.array([[1, 2, 3, 4,], [5, 6, 7, 8,]])
        >>> extend_all_channels(x_mat, R)
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
            Matrix of arrays to be centered.

    Returns
    -------
        numpy.ndarray
            Centered matrix array.

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
    Whiten the input matrix through zero component analysis.

    Parameters
    ----------
        x: numpy.ndarray
            Centred 2D array to be whitened.

    Returns
    -------
        numpy.ndarray
            Whitened array.

    Examples
    --------
        >>> x = np.array([[1, 2, 3, 4],  # Feature-1
                          [5, 6, 7, 8]]) # Feature-2
        >>> whiten(x)
        array([[0.6325, 1.265 , 1.8975, 2.53  ],
               [3.1625, 3.795 , 4.4275, 5.06  ]])
    """

    # Calculate covariance matrix
    cov_mat = np.cov(x, rowvar=True, bias=True)

    # Eigenvalues and eigenvectors
    w, v = linalg.eig(cov_mat)

    # Apply regularization factor, replacing eigenvalues smaller than it with the factor
    reg_factor = w[:round(len(w) / 2)].mean()
    w = np.where(w < reg_factor, reg_factor, w)

    # Diagonal matrix inverse square root of eigenvalues
    diagw = np.diag(1 / (w ** 0.5))
    diagw = diagw.real.round(4)

    # Whitening using zero component analysis: v diagw v.T x_cent
    wzca = np.dot(np.dot(np.dot(v, diagw), v.T), x)

    return wzca