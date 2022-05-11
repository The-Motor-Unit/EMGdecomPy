from scipy.io import loadmat
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
    raw_flattened = np.array([channel for channel in raw_flattened if 0 not in channel.shape]).squeeze()

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

    # Transpose array and cut off extra R rows
    extended_x = extended_x.T[0 : len(x)]

    return extended_x

def extend_input_all_channels(x_mat, R):
    """
    Takes an array with dimensions m by k,
    where m represents channels and k represents observations,
    and "extends" it to return an array of shape m by k by R+1.
    
    Parameters
    ----------
        x_mat: numpy.ndarray
            2D array to be extended.
        R: int
            How far to extend x.
        
    Returns
    -------
        numpy.ndarray
            m by k by R+1 extended array.
        
    Example:
        >>> R = 5
        >>> x_mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> extend_input_all_channels(x_mat, R)
        array([[[1., 0., 0., 0., 0., 0.],
                [2., 1., 0., 0., 0., 0.],
                [3., 2., 1., 0., 0., 0.]],

               [[4., 0., 0., 0., 0., 0.],
                [5., 4., 0., 0., 0., 0.],
                [6., 5., 4., 0., 0., 0.]]])

    """
    extended_x_mat = np.zeros([x_mat.shape[0], x_mat.shape[1], R + 1])
    for i, channel in enumerate(x_mat):
        
        # Extend channel
        extended_channel = extend_input_by_R(channel, R)

        # Add extended channel to the overall matrix of extended channels
        extended_x_mat[i] = extended_channel
        
    return extended_x_mat


def subtract_channel_average(raw):
    """
    Takes raw EMG signals and flattens matrix with
    flatten_signal function then finds the mean of each channel
    and subtracts it from the respective channel.
    
    Parameters
    ----------
        raw: numpy.ndarray
            Raw EMG signal array.
        
    Returns
    -------
        numpy.ndarray
            Raw EMG signals with means subtracted.
        
    Example
    -------
        >>> raw = np.array([[np.array([]),
                np.array([[2, 2, 8, 4, 5]]),
                np.array([[6, 17, 8, 29, 10]])
                        ]], dtype=object)
        >>> subtract_channel_average(raw)
        array([[-2.2, -2.2,  3.8, -0.2,  0.8],
               [-8. ,  3. , -6. , 15. , -4. ]])
    """
    
    raw_flattened = flatten_signal(raw)
    averages = []
    de_meaned = []
    for index, array in enumerate(raw_flattened):
        averages.append(array.mean())
        de_meaned.append(raw_flattened[index] - averages[index])
    return np.array(de_meaned)


SIG = np.array([[np.array([]),
         np.array([[2, 2, 8, 4, 5]]),
         np.array([[6, 17, 8, 29, 10]])
                ]], dtype=object)

subtract_channel_average(SIG)


