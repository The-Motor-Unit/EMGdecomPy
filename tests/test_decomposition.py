import emgdecompy as emg
import numpy as np
from scipy.io import loadmat
from scipy import linalg
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import variation
from sklearn.datasets import make_blobs


def test_initialize_w():
    """
    Run unit test on initialize_w function from EMGdecomPy.
    """

    x = np.array(
        [
            [1, 2, 3, 4],
            [5, 7, 9, 11],
            [12, 15, 18, 21],
        ]
    )
    assert (
        emg.decomposition.initialize_w(x) == np.array([4, 11, 21])
    ).all(), "Returned wrong column."

    x = np.zeros((5, 5))
    assert (
        emg.decomposition.initialize_w(x).shape == np.zeros(5).shape
    ), "Output contains wrong dimensions."


def test_normalize():
    """
    Run unit test on normalize function from EMGdecomPy.
    """
    for i in range(0, 10):

        # initialize array
        x = np.random.randint(1, 1000)
        y = np.random.randint(1, 1000)

        fake_data = np.random.rand(x, y)

        # calculate Frobenius norm manually
        squared = fake_data ** 2
        summed = squared.sum()
        frob_norm = np.sqrt(summed)

        # this is how the normalize() calculates Frobenius norm
        fx_norm = np.linalg.norm(fake_data)

        assert np.allclose(frob_norm, fx_norm), "Frobenius norm incorrectly calculated."

        # normalize
        normalized = fake_data / frob_norm

        fx = emg.decomposition.normalize(fake_data)

        assert np.allclose(normalized, fx), "Array normalized incorrectly."


def test_separation():
    """
    Run unit test on separation function from EMGdecomPy.
    """

    Tolx = 10e-4
    max_iter = 10

    # Call and process EMG data
    gl_10 = loadmat("data/raw/GL_10.mat")
    signal = gl_10["SIG"]
    # signal = create_emg_data(q=215473)

    x = emg.preprocessing.flatten_signal(signal)
    x = emg.preprocessing.center_matrix(x)
    x = emg.preprocessing.extend_all_channels(x, 16)
    z = emg.preprocessing.whiten(x)

    n = 0

    # Initialize separation vectors and matrix B
    w_curr = emg.decomposition.initialize_w(z)
    w_prev = emg.decomposition.initialize_w(z)
    B = np.zeros((1088, 1))

    while linalg.norm(np.dot(w_curr.T, w_prev) - 1) > Tolx:

        # Calculate separation vector
        b = np.dot(w_prev, z)
        g_b = emg.contrast.apply_contrast(b)
        A = (emg.contrast.apply_contrast(b, der=True)).mean()
        w_curr = (z * g_b).mean(axis=1) - A * w_prev

        # Orthogonalize and normalize separation vector
        w_curr = emg.decomposition.orthogonalize(w_curr, B)
        w_curr = emg.decomposition.normalize(w_curr)

        # Set previous separation vector to current separation vector
        w_prev = w_curr

        # If n exceeds max iteration exit while loop
        n += 1
        if n > max_iter:
            break

    assert (
        w_curr == emg.decomposition.separation(z, B, Tolx, emg.contrast.skew, max_iter)
    ).all(), "Separation vector incorrectly calculated."


def test_orthogonalize():
    """
    Run unit tests on orthogonalize() function from EMGdecomPy.
    """
    for i in range(0, 10):
        x = np.random.randint(1, 100)
        y = np.random.randint(1, 100)
        z = np.random.randint(1, 100)

        w = np.random.randn(x, y) * 1000
        b = np.random.randn(x, z) * 1000

        assert (
            b.T.shape[1] == w.shape[0]
        ), "Dimensions of input arrays are not compatible."

        # orthogonalize by hand
        ortho = w - b @ (b.T @ w)

        fx = emg.decomposition.orthogonalize(w, b)

        assert np.array_equal(
            ortho, fx
        ), "Manually calculated array not equivalent to emg.orthogonalize()"
        assert fx.shape == w.shape, "The output shape of w is incorrect."


def test_refinement(random_seed=42):
    """
    Run unit test on refinement function from EMGdecomPy.

        Parameters
    ----------
        random_seed: int
            used to randomize initial cv matrix and K-Means centers
    """
    gl_10 = loadmat("data/raw/GL_10.mat")
    signal = gl_10["SIG"]

    x = emg.preprocessing.flatten_signal(signal)
    x = emg.preprocessing.center_matrix(x)
    x = emg.preprocessing.extend_all_channels(x, 16)
    z = emg.preprocessing.whiten(x)
    B = np.zeros((1088, 1))
    Tolx = 10e-4

    w_i = emg.decomposition.separation(z, B, Tolx, emg.decomposition.skew, max_iter=10)
    max_iter = 10
    th_sil = 0.9

    np.random.seed(random_seed)
    cv_prev = np.random.ranf()
    cv_curr = cv_prev * 0.9

    # Get output of refine function
    w_i_ref = emg.decomposition.refinement(
        w_i, z, i=1, th_sil=0.9, filepath="", max_iter=10, random_seed=42
    )

    for iter in range(max_iter):

        # a. Estimate the i-th source
        s_i = np.dot(w_i, z)  # w_i and w_i.T are equal as far as I know

        # Estimate pulse train pt_n with peak detection applied to the square of the source vector
        s_i = np.square(s_i)

        peak_indices, _ = find_peaks(
            s_i, distance=41
        )  # 41 samples is ~equiv to 20 ms at a 2048 Hz sampling rate

        # b. Use KMeans to separate large peaks from relatively small peaks, which are discarded
        kmeans = KMeans(n_clusters=2, random_state=random_seed)
        kmeans.fit(s_i[peak_indices].reshape(-1, 1))
        centroid_a = np.argmax(
            kmeans.cluster_centers_
        )  # Determine which cluster contains large peaks
        peak_a = ~kmeans.labels_.astype(
            bool
        )  # Determine which peaks are large (part of cluster a)

        if centroid_a == 1:
            peak_a = ~peak_a

        peak_indices_a = peak_indices[
            peak_a
        ]  # Get the indices of the peaks in cluster a
        peak_indices_b = peak_indices[
            ~peak_a
        ]  # Get the indices of the peaks in cluster b

        # Create pulse train, where values are 0 except for when MU fires, which have values of 1
        pt_n = np.zeros_like(s_i)
        pt_n[peak_indices_a] = 1

        # c. Update inter-spike interval coefficients of variation
        J = len(peak_indices_a)
        cv_prev = cv_curr
        # to calculate difference per step
        # if your array is of length 3, eg [1, 4, 2]
        # you get isi of length 2 [3, -2]
        isi = np.zeros(J - 1)
        for step in range(J - 1):
            isi[step] = peak_indices_a[step + 1] - peak_indices_a[step]
        cv_curr = variation(isi)  # covariance of resulting steps

        if cv_curr > cv_prev:
            break

        # d. Updating vector via formula:
        # w_i(n + 1) = (1/J) * [sum{for j=1 to j=J} z.(t_j)]
        sum_cumul = np.zeros(z.shape[0])
        for i in range(J):
            peak_ind = peak_indices_a[i]
            sum_cumul = sum_cumul + z[:, peak_ind]

        w_i = sum_cumul * (1 / J)

    # If silhouette score is greater than threshold, accept estimated source and add w_i to B
    sil = emg.decomposition.silhouette_score(
        s_i, kmeans, peak_indices_a, peak_indices_b, centroid_a
    )

    if sil < th_sil:
        test_w_i = np.zeros_like(w_i)
    else:
        test_w_i = w_i

    # Check the dimensions of the output: expect it to be same as input array
    assert (
        w_i_ref.shape == test_w_i.shape
    ), "Shape of refined array does not match shape of input array"
    print(w_i_ref)
    print(test_w_i)
    assert np.allclose(
        w_i_ref, test_w_i
    ), "Different results for refined and manual array"


def test_pnr():
    """
    Run unit test on pnr function from EMGdecomPy.
    """
    samples = [3, 100, 1000]
    for i in samples:

        # create data with peaks
        features, clusters = make_blobs(
            n_samples=i, n_features=1, centers=2, random_state=50
        )
        features = features.flatten() ** 2

        peak_indices, _ = find_peaks(features)

        fx_pnr = emg.decomposition.pnr(features, peak_indices)

        # calculate pnr by hand
        pulse = []
        noise = []

        for j in features:

            condition = j in list(features[peak_indices])

            if condition == True:
                pulse.append(j)  # numerator
            else:
                noise.append(j)  # denominator

        # calculate average height
        pulse = np.array(pulse).mean()
        noise = np.array(noise).mean()

        p_to_n = pulse / noise

        assert fx_pnr == p_to_n, "PNR incorrectly calculated."


def test_silhouette_score():
    """
    Run unit test on silhouette_score function from EMGdecomPy.
    """
    sil_by_hand = [0.96296095, 0.89659568]  # sil scores calculated by hand
    state_list = [50, 250]

    for i, state in enumerate(state_list):

        features, clusters = make_blobs(
            n_samples=10, n_features=1, centers=3, random_state=state
        )
        features = features.flatten() * 100  # flatten to find_peaks

        peak_indices, _ = find_peaks(features)
        peak_indices

        peak_features = features[peak_indices].reshape(-1, 1)

        kmeans = KMeans(n_clusters=2).fit(peak_features)
        center = kmeans.cluster_centers_

        # taken from refinement fx
        centroid_a = np.argmax(
            kmeans.cluster_centers_
        )  # Determine which cluster contains large peaks
        peak_a = ~kmeans.labels_.astype(
            bool
        )  # Determine which peaks are large (part of cluster a)

        if centroid_a == 1:
            peak_a = ~peak_a

        peak_indices_a = peak_indices[
            peak_a
        ]  # Get the indices of the peaks in cluster a
        peak_indices_b = peak_indices[
            ~peak_a
        ]  # Get the indices of the peaks in cluster b
        peak_features = features[peak_indices].reshape(-1, 1)

        # irl this needs to be emg.decomposition.silhouette_score()
        sil = emg.decomposition.silhouette_score(
            features, kmeans, peak_indices_a, peak_indices_b, centroid_a
        )

        assert np.isclose(
            sil, sil_by_hand[i]
        ), "Inter and intra distances incorrectly calculated."
