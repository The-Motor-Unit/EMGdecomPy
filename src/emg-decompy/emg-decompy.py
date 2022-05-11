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


def subtract_channel_average(raw_flattened, extended_x_mat):
    """
    Takes a flattened matrix from flatten_signal function,
    finds the mean of each channel, and subtracts it from 
    each channel in the extended x matrix given by function
    extend_all_input_channels.
    
    Parameters
    ----------
        raw_flattened: numpy.ndarray
            2D array to find averages.
        extended_x_mat: numpy.ndarray
            array to subtract averages from.
        
    Returns
    -------
        numpy.ndarray
            extended_x_mat with averages subtracted.
        
    Example:
        >>> raw_flattened = np.array([[np.array([]),
                                         np.array([[46.79361979, 13.22428385, -7.12076823, 26.44856771,
                                                 50.86263021, 36.62109375]])                             ,
                                         np.array([[76.29394531, 39.67285156, 29.50032552, 34.58658854,
                                                 47.8108724 , 57.98339844]])                             ,
                                         np.array([[74.2594401 ,  4.06901042, -5.08626302, 21.36230469,
                                                 30.51757812, 36.62109375]])                             ,
                                         np.array([[62.05240885, 28.48307292, 17.29329427,  3.05175781,
                                                 22.37955729, -5.08626302]])                             ]],
                                                 dtype=object)
        >>> extended_x_mat = array([[[ 1.,  0.,  0.,  0.,  0.,  0.],
                                    [ 2.,  1.,  0.,  0.,  0.,  0.],
                                    [ 3.,  2.,  1.,  0.,  0.,  0.]],

                                   [[ 4.,  0.,  0.,  0.,  0.,  0.],
                                    [ 5.,  4.,  0.,  0.,  0.,  0.],
                                    [ 6.,  5.,  4.,  0.,  0.,  0.]],

                                   [[ 7.,  0.,  0.,  0.,  0.,  0.],
                                    [ 8.,  7.,  0.,  0.,  0.,  0.],
                                    [ 9.,  8.,  7.,  0.,  0.,  0.]],

                                   [[10.,  0.,  0.,  0.,  0.,  0.],
                                    [11., 10.,  0.,  0.,  0.,  0.],
                                    [12., 11., 10.,  0.,  0.,  0.]]])

    """
    averages = []
    for index, array in enumerate(raw_flattened):
        averages.append(array.mean())
    for channel, arr in enumerate(extended_x_mat):
        for ext_index, ext_array in enumerate(extended_x_mat[channel]):
            extended_x_mat[channel][ext_index] = extended_x_mat[channel][ext_index] - np.averages[channel]
    return np.array(extended_x_mat)
    


SIG = np.array([[np.array([]),
         np.array([[46.79361979, 13.22428385, -7.12076823, 26.44856771,
                 50.86263021, 36.62109375]])                             ,
         np.array([[76.29394531, 39.67285156, 29.50032552, 34.58658854,
                 47.8108724 , 57.98339844]])                             ,
         np.array([[74.2594401 ,  4.06901042, -5.08626302, 21.36230469,
                 30.51757812, 36.62109375]])                             ,
         np.array([[62.05240885, 28.48307292, 17.29329427,  3.05175781,
                 22.37955729, -5.08626302]])                             ]], dtype=object)

x = flatten_signal(SIG)

extended = extend_input_all_channels(np.array([[1, 2, 3], [4, 5, 6],[7,8,9],[10,11,12]]), 5)

subtract_channel_average(x, extended)


