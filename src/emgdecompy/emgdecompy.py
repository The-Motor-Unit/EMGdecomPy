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
    Whiten the input matrix.
    First, the data is centred by subtracting the mean and then ZCA whitening is performed.

    Parameters
    ----------
        x: numpy.ndarray
            2D array to be whitened.

    Returns
    -------
        numpy.ndarray
            Whitened array.

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
    Takes first derivitive and applies contrast function to w with map()
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


def refinement(w_i, z, i, th_sil=0.9, filepath="", max_iter=10):
    """
    Refines the estimated separation vectors
    determined by the fixed point algorithm as described in Negro et al. (2016).
    Uses a peak-finding algorithm combined with K-Means clustering
    to determine the motor unit pulse train.

    Parameters
    ----------
        w_i: numpy.ndarray
            Current separation vector to refine.
        z: numpy.ndarray
            Centred, extended, and whitened EMG data.
        i: int
            Decomposition iteration number.
        max_iter: int > 0
            Maximum iterations for refinement.
        th_sil: float
            Silhouette score threshold for accepting a separation vector.
        filepath: str
            Filepath/name to be used when saving pulse trains.

    Returns
    -------
        numpy.ndarray
            Separation vector if silhouette score is below threshold.
            Otherwise return nothing.

    Examples
    --------
    >>> w_i = refinement(w_i, z) # where z in extended, whitened, centered emg data
    """
    # Initialize inter-spike interval coefficient of variations for n and n-1 as random numbers
    cv_curr, cv_prev = np.random.ranf(), np.random.ranf()

    if cv_curr > cv_prev:
        cv_curr, cv_prev = cv_prev, cv_curr

    n = 0

    while cv_curr < cv_prev:

        # a. Estimate the i-th source
        s_i = np.dot(w_i, z)  # w_i and w_i.T are equal as far as I know

        # Estimate pulse train pt_n with peak detection applied to the square of the source vector
        s_i = np.square(s_i)

        peak_indices, _ = find_peaks(
            s_i, distance=41
        )  # 41 samples is ~equiv to 20 ms at a 2048 Hz sampling rate

        # b. Use KMeans to separate large peaks from relatively small peaks, which are discarded
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(peak_indices)
        centroid_a = np.argmax(
            kmeans.cluster_centers_
        )  # Determine which cluster contains large peaks
        peak_a = ~kmeans.labels_.astype(
            bool
        )  # Determine which peaks are large (part of cluster a)

        if centroid_a == 1:
            peak_a = ~peak_a

        peak_a = peak_indices[peak_a]  # Get the indices of the peaks in cluster a

        # Create pulse train, where values are 0 except for when MU fires, which have values of 1
        pt_n = np.zeros_like(s_i)
        pt_n[peak_a] = 1

        # c. Update inter-spike interval coefficients of variation
        isi = np.diff(peak_a)  # inter-spike intervals
        cv_prev = cv_curr
        cv_curr = variation(isi)

        # d. Update separation vector
        j = len(peak_a)

        w_i = (1 / j) * z[:, peak_a].sum(axis=1)

        n += 1

    # Save pulse train
    pd.DataFrame(pt_n, columns=["pulse_train"]).rename_axis("sample").to_csv(
        f"{filepath}_PT_{i}"
    )

    # If silhouette score is greater than threshold, accept estimated source and add w_i to B
    sil = silhouette_score(
        peak_indices, kmeans.labels_
    )  # Definition is slightly different than in paper, may change

    if sil < th_sil:
        return  # If below threshold, reject estimated source and return nothing

    return w_i  # May change implementation to update B here


def decomposition(
    x, M=64, Tolx=10e-4, fun=skew, max_iter_sep=10, th_sil=0.9, filepath="", max_iter_ref=10
):
    """
    Main function duplicating decomposition algorithm from Negro et al. (2016).
    Performs decomposition of raw EMG signals.

    Parameters
    ----------
        x: numpy.ndarray
            The input matrix.
        Tolx: float
            Tolx for element-wise comparison in separation.
        fun: function
            Contrast function to use.
            skew, og_cosh or exp_sq
        max_iter_sep: int > 0
            Maximum iterations for fixed point algorithm.
            When to stop if it doesn't converge.
        th_sil: float
            Silhouette score threshold for accepting a separation vector.
        max_iter_ref: int > 0
            Maximum iterations for refinement.
        filepath: str
            Filepath/name to be used when saving pulse trains.

    Returns
    -------
        numpy.ndarray
            Decomposed matrix B.

    Examples
    --------
    >>> x = gl_10 = loadmat('../data/raw/gl_10.mat') #Classic gold standard data
    >>> x = gl_to['SIG']
    >>> decomposition(x)

    """
    # Flatten
    x = flatten_signal(x)

    # Extend
    x_ext = extend_all_channels(x, 10)

    # Subtract mean + Whiten
    z = whiten(x_ext)

    B = np.zeros((z.shape[0], z.shape[0]))

    for i in range(M):

        # Separate
        w_i = separation(z, B, Tolx, fun, max_iter_sep)

        # Refine
        B[:i] = refinement(w_i, z, i, max_iter_ref, th_sil, filepath)

    return B
