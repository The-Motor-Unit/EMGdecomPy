import numpy as np
import pandas as pd
from emgdecompy.preprocessing import flatten_signal, extend_all_channels, whiten
from emgdecompy.contrast import skew, apply_contrast
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import variation


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
        # w_curr = np.dot(z, w_curr).mean()
        w_curr = (z * w_curr).mean(axis=1)
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
    x,
    M=64,
    Tolx=10e-4,
    fun=skew,
    max_iter_sep=10,
    th_sil=0.9,
    filepath="",
    max_iter_ref=10,
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
