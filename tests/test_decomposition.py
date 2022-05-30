import emgdecompy as emg
import numpy as np
from scipy.io import loadmat
from scipy import linalg
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import variation
from test_preprocessing import create_emg_data


def test_initialize_w():
    """
    Run unit test on initialize_w function from EMGdecomPy.
    """
    x = np.array(
        [
            [
                1,
                2,
                3,
                4,
            ],
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


def test_refinement():
    """
    Run unit test on refinement function from EMGdecomPy.

    Parameters of test_refinement() function:
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
    """

    # Call and process EMG data
    gl_10 = loadmat("data/raw/GL_10.mat")
    signal = gl_10["SIG"]

    x = emg.preprocessing.flatten_signal(signal)
    x = emg.preprocessing.center_matrix(x)
    x = emg.preprocessing.extend_all_channels(x, 16)
    z = emg.preprocessing.whiten(x)
    B = np.zeros((1088, 1))
    Tolx = 10e-4

    w_i = emg.decomposition.separation(z, B, Tolx, emg.contrast.skew, max_iter=10)
    w_init = w_i  # Preserve for comparison tests
    max_iter = 10
    th_sil = 0.9

    np.random.seed(42)
    cv_prev = np.random.ranf()
    cv_curr = cv_prev * 0.9

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
        np.random.seed(42)
        kmeans = KMeans(n_clusters=2)
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
        isi = np.diff(peak_indices_a)  # inter-spike intervals
        cv_prev = cv_curr
        cv_curr = variation(isi)

        # d. Update separation vector
        j = len(peak_indices_a)

        w_i = (1 / j) * z[:, peak_indices_a].sum(axis=1)

        n += 1

        if n == max_iter:
            break

    # If silhouette score is greater than threshold, accept estimated source and add w_i to B
    sil = silhouette_score(s_i, kmeans, peak_indices_a, peak_indices_b, centroid_a)

    if sil < th_sil:
        res = np.zeros_like(
            w_i
        )  # If below threshold, reject estimated source and return nothing
    else:
        res = w_i

    # Via refinement function
    w_i_ref = refinement(
        w_init,
        z,
        i,
        th_sil=0.9,
        filepath="",
        max_iter=10,
        use_rand_seed=True,
        rand_seed=42,
    )

    # Check the dimensions of the output: expect it to be same as input array
    assert (
        w_i_ref.shape == res.shape
    ), "Shape of refined array does not match shape of input array"

    assert np.allclose(res, w_i_ref), "Different results for refined and manual array"
