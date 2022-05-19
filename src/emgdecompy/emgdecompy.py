from scipy.io import loadmat
from scipy import linalg
import pandas as pd
import altair as alt
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import variation

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
    extended_x_mat = np.zeros([x_mat.shape[0],  (R + 1), x_mat.shape[1]])

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

def refinement(w_i, z):
    # Initialize inter-spike interval coefficient of variations for n and n-1 as random numbers
    cv_curr, cv_prev = np.random.ranf(), np.random.ranf()

    if cv_curr > cv_prev:
        cv_curr, cv_prev = cv_prev, cv_curr 

    n = 0

    while cv_curr < cv_prev:
        
        # a. Estimate the i-th source 
        s_i = np.dot(w_i, z) # w_i and w_i.T are equal as far as I know
        
        # Estimate pulse train pt_n with peak detection applied to the square of the source vector
        
        s_i = np.square(s_i)
        
        peak_indices, _ = find_peaks(s_i, distance=41) #41 samples is ~equiv to 20 ms at a 2048 Hz sampling rate
        
        # b. Use KMeans to separate large peaks from relatively small peaks, which are discarded
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(peak_indices)
        clust_a = np.argmax(kmeans.cluster_centers_)
        peak_a = ~kmeans.labels_.astype(bool) # Determine which peaks are large
        
        if clust_a == 1:
            peak_a = ~peak_a
        
        peak_a = peak_indices[peak_a]
        
        # Create pulse train, where values are 0 except for when MU fires, which have values of 1
        pt_n = np.zeros_like(s_i)
        pt_n[peak_a] = 1
        
        # c. Update inter-spike interval coefficients of variation
        isi = np.diff(peak_a) #inter-spike intervals
        cv_prev = cv_curr
        cv_curr = variation(isi)
        
        # d. Update separation vector
        j = len(peak_a)
        
        w_i =  (1 / j) * z[:, peak_a].sum(axis=1)
        
        n += 1
        
    # Add way to save pulse train

    # If silhouette score is greater than threshold, accept estimated source and add w_i to B   
    sil = silhouette_score(peak_indices, kmeans.labels_) # Definition is slightly different than in paper, may change

    if sil < th_sil:
        w_i = np.zeros_like(w_i)

    return w_i # May change implementation to update B here 