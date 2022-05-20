import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import variation

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
