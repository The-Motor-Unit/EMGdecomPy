# Copyright (C) 2022 Daniel King, Jasmine Ortega, Rada Rudyak, Rowan Sivanandam
# This script contains functions used to run the blind source separation algorithm
# based off the work in Francesco Negro et al 2016 J. Neural Eng. 13 026027.

import numpy as np
from emgdecompy.preprocessing import flatten_signal, butter_bandpass_filter, center_matrix, extend_all_channels, whiten
from emgdecompy.contrast import skew, apply_contrast
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from scipy.stats import variation


def initial_w_matrix(z, l=31):
    """
    Find highest activity regions of z to use as initializations of w. 
    Highest activity regions of z refers to the time instances corresponding
    to the highest values in the squared summation of all the whitened and
    extended observation vectors. Used for step 1 in Negro et al. 2016.

    Parameters
    ----------
        z: numpy.ndarray
            The whitened extended observation matrix.
            shape = M*(R+1) x K
            M = number of channels
            R = extension factor
            K = number of time points
        l: int
            Required minimal horizontal distance between peaks.
            Default value of 31 samples is approximately equivalent
            to 15 ms at a 2048 Hz sampling rate.

    Returns
    -------
        numpy.ndarray
            Peak indices for high activity columns of z.
        numpy.ndarray
            Corresponding  peak heights for each peak index.

    Examples
    --------
    >>> initial_w_matrix(z)
    """

    z_summed = np.sum(z, axis=0)  # sum across rows. shape = 1 x K
    z_squared = z_summed ** 2  # square each value. shape = 1 x K

    z_peak_indices, z_peak_info = find_peaks(z_squared, distance=l, height=0)
    z_peak_heights = z_peak_info["peak_heights"]
    
    return z_peak_indices, z_peak_heights


def deflate(w, B):
    """
    w = w - BB^{T} * w
    Note: this is not true orthogonalization, such as the Gram–Schmidt process.
    This is dubbed in Negro et al. (2016) as the "source deflation procedure."

    Parameters
    ----------
        w: numpy.ndarray
            Vector we are "orthogonalizing" against columns of B.
        B: numpy.ndarray
            Matrix of vectors to "orthogonalize" w by. 
            Should contain float dtype.

    Returns
    -------
        numpy.ndarray
            'Deflated' w.

    Examples
    --------
        >>> w = np.array([7, 4, 6])
        >>> B = np.array([[ 1. ,  1.2,  0. ],
                          [ 2. , -0.6,  0. ],
                          [ 0. ,  0. ,  0. ]])
        >>> deflate(w, B)
        array([-28. ,  -3.2,   6. ])
    """
    w = w - np.dot(B.T, np.dot(B, w))
    return w

def gram_schmidt(w, B):
    """
    Gram-Schmidt orthogonalization.

    Parameters
    ----------
        w: numpy.ndarray
            Vector we are orthogonalizing against columns of B.
        B: numpy.ndarray
            Matrix of vectors to orthogonalize w by. 
            Should contain float dtype.

    Returns
    -------
        numpy.ndarray
            Orthogonalized w.

    Examples
    --------
        >>> w = np.array([7, 4, 6])
        >>> B = np.array([[ 1. ,  1.2,  0. ],
                          [ 2. , -0.6,  0. ],
                          [ 0. ,  0. ,  0. ]])
        >>> gram_schmidt(w, B)
        array([0., 0., 6.])
    """
    projw_a = 0
    for i in range(B.shape[1]):
        a = B[:, i]
        if np.all(a == 0):
            continue
        projw_a = projw_a + (np.dot(w, a) / np.dot(a, a)) * a
    w = w - projw_a
    return w

def orthogonalize(w, B, fun=gram_schmidt):
    """
    Performs orthogonalization using selected orthogonalization function.
    
     Parameters
    ----------
        w: numpy.ndarray
            Vector we are orthogonalizing against columns of B.
        B: numpy.ndarray
             Matrix of vectors to orthogonalize w by. 
             Should contain float dtype.
        fun: function
            What function to use for orthogonalizing process.
            Current options are:
                - gram_schmidt (default)
                - deflate

    Returns
    -------
        numpy.ndarray
            Orthogonalized w.

    Examples
    --------
        >>> w = np.array([7, 4, 6])
        >>> B = np.array([[ 1. ,  1.2,  0. ],
                          [ 2. , -0.6,  0. ],
                          [ 0. ,  0. ,  0. ]])
        >>> orthogonalize(w, B)
        array([0., 0., 6.])
    """
    return fun(w,B)

def normalize(w):
    """
    Normalize the input vector (scale the elements of the vector so  its length is 1).
    This is done using the formula `w/||w||`.

    Parameters
    ----------
        w: numpy.ndarray
            Vector to normalize.

    Returns
    -------
        numpy.ndarray
            Normalized vector.

    Examples
    --------
        >>> w = np.array([5, 6, 23, 29])
        >>> normalize(w)
        array([0.13217526, 0.15861032, 0.60800622, 0.76661653])

    """
    norms = np.linalg.norm(w)
    w = w / norms
    return w


def separation(
    z,
    w_init,
    B,
    Tolx=10e-4,
    contrast_fun=skew,
    ortho_fun=gram_schmidt,
    max_iter=10,
    verbose=False,
):
    """
    Finds the separation vector for the i-th source using latent component analysis
    that maximizes for sparsity. Implemented with a fixed point algorithm.
    Step 2 in Negro et al.(2016).

    Parameters
    ----------
        z: numpy.ndarray
            Extended and whitened observation matrix.
        w_init: numpy.ndarray
            Initial separation vector.
        B: numpy.ndarray
            Current separation matrix.
        Tolx: numpy.ndarray
            Tolx for element-wise comparison.
        contrast_fun: function
            Contrast function to use.
            skew, log_cosh or exp_sq
        ortho_fun: function
            Orthogonalization function to use.
            gram_schmidt or deflate or None
        max_iter: int > 0
            Maximum iterations for fixed point algorithm.
            When to stop if it doesn't converge.
        verbose: bool
            If true, print fixed-point algorithm iterations.

    Returns
    -------
        numpy.ndarray
            Estimated separation vector for the i-th source.

    Examples
    --------
    >>> w_i = separation(z, w_init, B)

    """
    n = 0
    w_curr = w_init
    w_prev = w_curr

    while np.linalg.norm(np.dot(w_curr.T, w_prev) - 1) > Tolx and n < max_iter:

        w_prev = w_curr

        # -------------------------
        # 2a: Fixed point algorithm
        # -------------------------

        # Calculate A
        # A = average of (der of contrast function (transposed prev(w) x z))
        # A = E{g'[w_prev{T}.z]}
        A = np.dot(w_prev.T, z)
        A = apply_contrast(A, contrast_fun, True).mean()

        # Calculate new w_curr
        w_curr = np.dot(w_prev.T, z)
        w_curr = apply_contrast(w_curr, contrast_fun, False)
        w_curr = (z * w_curr).mean(axis=1) # Same as taking dot product and dividing by number of data points
        w_curr = w_curr - A * w_prev

        # -------------------------
        # 2b: Orthogonalize
        # -------------------------
        if ortho_fun != None: # Don't orthogonalize if ortho_fun is None
            w_curr = orthogonalize(w_curr, B, ortho_fun)

        # -------------------------
        # 2c: Normalize
        # -------------------------
        w_curr = normalize(w_curr)

        # -------------------------
        # 2d: Iterate
        # -------------------------
        n = n + 1

    if n < max_iter and verbose:
        print(f"Fixed-point algorithm converged after {n} iterations.")

    return w_curr


def silhouette_score(s_i, peak_indices):
    """
    Calculates silhouette score on the estimated source.

    Defined as the difference between within-cluster sums of point-to-centroid distances
    and between-cluster sums of point-to-centroid distances.
    Measure is normalized by dividing by the maximum of these two values (Negro et al. 2016).

    Parameters
    ----------
        s_i: numpy.ndarray
            Estimated source. 1D array containing K elements, where K is the number of samples.
        peak_indices_a: numpy.ndarray
            1D array containing the peak indices.

    Returns
    -------
        float
            Silhouette score.

    Examples
    --------
    >>> s_i = np.array([0.80749775, 10, 0.49259282, 0.88726069, 5,
                        0.86282998, 3, 0.79388539, 0.29092294, 2])
    >>> peak_indices = np.array([1, 4, 6, 9])
    >>> silhouette_score(s_i, peak_indices)
    0.740430148513959

    """
    # Create clusters
    peak_cluster = s_i[peak_indices]
    noise_cluster = np.delete(s_i, peak_indices)

    # Create centroids
    peak_centroid = peak_cluster.mean()
    noise_centroid = noise_cluster.mean()

    # Calculate within-cluster sums of point-to-centroid distances
    intra_sums = (
        abs(peak_cluster - peak_centroid).sum()
        + abs(noise_cluster - noise_centroid).sum()
    )

    # Calculate between-cluster sums of point-to-centroid distances
    inter_sums = (
        abs(peak_cluster - noise_centroid).sum()
        + abs(noise_cluster - peak_centroid).sum()
    )

    diff = inter_sums - intra_sums

    sil = diff / max(intra_sums, inter_sums)

    return sil


def pnr(s_i, peak_indices):
    """
    Returns pulse-to-noise ratio of an estimated source.
    
    Parameters
    ----------
    s_i: numpy.ndarray
        Square of estimated source. 1D array containing K elements, where K is the number of samples.
    peak_indices: numpy.ndarray
        1D array containing the peak indices.
    
    Returns
    -------
        float
            Pulse-to-noise ratio.
            
    Examples
    --------
    >>> s_i = np.array([0.80749775, 10, 0.49259282, 0.88726069, 5,
                        0.86282998, 3, 0.79388539, 0.29092294, 2])
    >>> peak_indices = np.array([1, 4, 6, 9])
    >>> pnr(s_i, peak_indices)
    8.606468362838562
    """
    
    signal = 10 * np.log10(s_i[peak_indices].mean())
    noise = 10 * np.log10(np.delete(s_i, peak_indices).mean())
    
    return signal - noise


def refinement(
    w_i, z, i, l=31, sil_pnr=True, thresh=0.9, max_iter=10, random_seed=None, verbose=False
):
    """
    Refines the estimated separation vectors determined by the `separation` function
    as described in Negro et al. (2016). Uses a peak-finding algorithm combined
    with K-Means clustering to determine the motor unit spike train. Updates the 
    estimated separation vector accordingly until regularity of the spike train is
    maximized. Steps 4, 5, and 6 in Negro et al. (2016).

    Parameters
    ----------
        w_i: numpy.ndarray
            Current separation vector to refine.
        z: numpy.ndarray
            Centred, extended, and whitened EMG data.
        i: int
            Decomposition iteration number.
        l: int
            Required minimal horizontal distance between peaks in peak-finding algorithm.
            Default value of 31 samples is approximately equivalent
            to 15 ms at a 2048 Hz sampling rate.
        sil_pnr: bool
            Whether to use SIL or PNR as acceptance criterion.
            Default value of True uses SIL.
        thresh: float
            SIL/PNR threshold for accepting a separation vector.
        max_iter: int > 0
            Maximum iterations for refinement.
        random_seed: int
            Used to initialize the pseudo-random processes in the function.
        verbose: bool
           If true, refinement information is printed.

    Returns
    -------
        numpy.ndarray
            Separation vector if SIL/PNR is above threshold.
            Otherwise return empty vector.
        numpy.ndarray
            Estimated source obtained from dot product of separation vector and z.
            Empty array if separation vector not accepted.
        numpy.ndarray
            Peak indices for peaks in cluster "a" of the squared estimated source.
            Empty array if separation vector not accepted.
        float
            Silhouette score if SIL/PNR is above threshold.
            Otherwise return 0.
        float
            Pulse-to-noise ratio if SIL/PNR is above threshold.
            Otherwise return 0.

    Examples
    --------
    >>> w_i = refinement(w_i, z, i)
    """
    cv_curr = np.inf # Set it to inf so there isn't a chance the loop breaks too early

    for iter in range(max_iter):
        
        w_i = normalize(w_i) # Normalize separation vector

        # a. Estimate the i-th source
        s_i = np.dot(w_i, z)  # w_i and w_i.T are equal

        # Estimate pulse train pt_n with peak detection applied to the square of the source vector
        s_i2 = np.square(s_i)

        # Peak-finding algorithm
        peak_indices, _ = find_peaks(
            s_i2, distance=l
        )

        # b. Use KMeans to separate large peaks from relatively small peaks, which are discarded
        kmeans = KMeans(n_clusters=2, random_state=random_seed)
        kmeans.fit(s_i2[peak_indices].reshape(-1, 1))
        
        # Determine which cluster contains large peaks
        centroid_a = np.argmax(
            kmeans.cluster_centers_
        )
        
        # Determine which peaks are large (part of cluster a)
        peak_a = ~kmeans.labels_.astype(
            bool
        )

        if centroid_a == 1: # If cluster a corresponds to kmeans label 1, change indices correspondingly
            peak_a = ~peak_a

        
        # Get the indices of the peaks in cluster a
        peak_indices_a = peak_indices[
            peak_a
        ]

        # c. Update inter-spike interval coefficients of variation
        isi = np.diff(peak_indices_a)  # inter-spike intervals
        cv_prev = cv_curr
        cv_curr = variation(isi)

        if np.isnan(cv_curr): # Translate nan to 0
            cv_curr = 0

        if (
            cv_curr > cv_prev
        ):
            break
            
        elif iter != max_iter - 1: # If we are not on the last iteration
            # d. Update separation vector for next iteration unless refinement doesn't converge
            j = len(peak_indices_a)
            w_i = (1 / j) * z[:, peak_indices_a].sum(axis=1)

    # If silhouette score is greater than threshold, accept estimated source and add w_i to B
    sil = silhouette_score(
        s_i2, peak_indices_a
    )
    pnr_score = pnr(s_i2, peak_indices_a)
    
    if isi.size > 0 and verbose:
        print(f"Cov(ISI): {cv_curr / isi.mean() * 100}")

    if verbose:
        print(f"PNR: {pnr_score}")
        print(f"SIL: {sil}")
        print(f"cv_curr = {cv_curr}")
        print(f"cv_prev = {cv_prev}")
        
        if cv_curr > cv_prev:
            print(f"Refinement converged after {iter} iterations.")

    if sil_pnr:
        score = sil # If using SIL as acceptance criterion
    else:
        score = pnr_score # If using PNR as acceptance criterion
    
    # Don't accept if score is below threshold or refinement doesn't converge
    if score < thresh or cv_curr < cv_prev or cv_curr == 0: 
        w_i = np.zeros_like(w_i) # If below threshold, reject estimated source and return nothing
        return w_i, np.zeros_like(s_i), np.array([]), 0, 0
    else:
        print(f"Extracted source at iteration {i}.")
        return w_i, s_i, peak_indices_a, sil, pnr_score


def decomposition(
    x,
    discard=None,
    R=16,
    M=64,
    bandpass=True,
    lowcut=10,
    highcut = 900,
    fs=2048,
    order=6,
    Tolx=10e-4,
    contrast_fun=skew,
    ortho_fun=gram_schmidt,
    max_iter_sep=10,
    l=31,
    sil_pnr=True,
    thresh=0.9,
    max_iter_ref=10,
    random_seed=None,
    verbose=False
):
    """
    Blind source separation algorithm that utilizes the functions
    in EMGdecomPy to decompose raw EMG data. Runs data pre-processing, separation,
    and refinement steps to extract individual motor unit activity from EMG data. 
    Runs steps 1 through 6 in Negro et al. (2016).

    Parameters
    ----------
        x: numpy.ndarray
            Raw EMG signal.
        discard: slice, int, or array of ints
            Indices of channels to discard.
        R: int
            How far to extend x.
        M: int
            Number of iterations to run decomposition for.
        bandpass: bool
            Whether to band-pass filter the raw EMG signal or not.
        lowcut: float
            Lower range of band-pass filter.
        highcut: float
            Upper range of band-pass filter.
        fs: float
            Sampling frequency in Hz.
        order: int
            Order of band-pass filter. 
        Tolx: float
            Tolerance for element-wise comparison in separation.
        contrast_fun: function
            Contrast function to use.
            skew, og_cosh or exp_sq
        ortho_fun: function
            Orthogonalization function to use.
            gram_schmidt or deflate
        max_iter_sep: int > 0
            Maximum iterations for fixed point algorithm.
        l: int
            Required minimal horizontal distance between peaks in peak-finding algorithm.
            Default value of 31 samples is approximately equivalent
            to 15 ms at a 2048 Hz sampling rate.
        sil_pnr: bool
            Whether to use SIL or PNR as acceptance criterion.
            Default value of True uses SIL.
        thresh: float
            SIL/PNR threshold for accepting a separation vector.
        max_iter_ref: int > 0
            Maximum iterations for refinement.
        random_seed: int
            Used to initialize the pseudo-random processes in the function.
        verbose: bool
            If true, decomposition information is printed.

    Returns
    -------
        dict
            Dictionary containing:
                B: numpy.ndarray
                    Matrix whose columns contain the accepted separation vectors.
                MUPulses: numpy.ndarray
                    Firing indices for each motor unit.
                SIL: numpy.ndarray
                    Corresponding silhouette scores for each accepted source.
                PNR: numpy.ndarray
                    Corresponding pulse-to-noise ratio for each accepted source.

    Examples
    --------
    >>> gl_10 = loadmat('../data/raw/gl_10.mat')
    >>> x = gl_10['SIG']
    >>> decomposition(x)
    """

    # Flatten
    x = flatten_signal(x)
    
    # Discard unwanted channels
    if discard != None:
        x = np.delete(x, discard, axis=0)

    # Apply band-pass filter
    if bandpass:
        x = np.apply_along_axis(
            butter_bandpass_filter,
            axis=1,
            arr=x,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs, 
            order=order
        )

    # Center
    x = center_matrix(x)

    print("Centred.")

    # Extend
    x_ext = extend_all_channels(x, R)

    print("Extended.")

    # Whiten
    z = whiten(x_ext)

    print("Whitened.")

    decomp_results = {}  # Create output dictionary

    B = np.zeros((z.shape[0], z.shape[0]))  # Initialize separation matrix
    
    z_peak_indices, z_peak_heights = initial_w_matrix(z)  # Find highest activity columns in z
    z_peaks = z[:, z_peak_indices] # Index the highest activity columns in z

    MUPulses = []
    sils = []
    pnrs = []

    for i in range(M):

        z_highest_peak = (
            z_peak_heights.argmax()
        )  # Determine which column of z has the highest activity

        w_init = z_peaks[
            :, z_highest_peak
        ]  # Initialize the separation vector with this column

        if verbose and (i + 1) % 10 == 0:
            print(i)

        # Separate
        w_i = separation(
            z, w_init, B, Tolx, contrast_fun, ortho_fun, max_iter_sep, verbose
        )

        # Refine
        w_i, s_i, mu_peak_indices, sil, pnr_score = refinement(
            w_i, z, i, l, sil_pnr, thresh, max_iter_ref, random_seed, verbose
        )
    
        B[:, i] = w_i # Update i-th column of separation matrix

        if mu_peak_indices.size > 0:  # Only save information for accepted vectors
            MUPulses.append(mu_peak_indices)
            sils.append(sil)
            pnrs.append(pnr_score)

        # Update initialization matrix for next iteration
        z_peaks = np.delete(z_peaks, z_highest_peak, axis=1)
        z_peak_heights = np.delete(z_peak_heights, z_highest_peak)
        
    decomp_results["B"] = B[:, B.any(0)] # Only save columns of B that have accepted vectors
    decomp_results["MUPulses"] = np.array(MUPulses, dtype="object")
    decomp_results["SIL"] = np.array(sils)
    decomp_results["PNR"] = np.array(pnrs)

    return decomp_results
