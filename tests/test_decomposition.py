import emgdecompy as emg
import numpy as np
import pytest

from scipy.io import loadmat
from scipy import linalg
from scipy.signal import find_peaks
from scipy.stats import variation
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Test script for functions defined in src/decomposition.py

# Note: Fixtures are special pytest objects calld into individual tests,
# they are useful when data is required to test a function


@pytest.fixture
def Z():
    """
    Create subset of EMG data to test with.
    """
    # load data
    gl_10 = loadmat("data/raw/GL_10.mat")
    raw = gl_10["SIG"]

    # select two channels from raw data
    data = raw[1, 1:3]

    # flatten
    x = emg.preprocessing.flatten_signal(data)

    # extend
    x_ext = emg.preprocessing.extend_all_channels(x, 10)

    # whiten data
    z = emg.preprocessing.whiten(x_ext)
    return z


def test_initialize_w(Z):
    """
    Run unit test on initialize_w function from EMGdecomPy.
    """
    # manually set 3 high values within range of l
    high_idx = [10, 120]

    for k in high_idx:

        Z[0][k] = np.max(Z) * 1000
        Z[0][k + 1] = np.max(Z) * 1000
        Z[0][k + 2] = np.max(Z) * 1000

        # set l to low distance
        l = 2

        idx, heights = emg.decomposition.initial_w_matrix(Z, l=l)

        # find heighest peak identified
        i = np.argmax(heights)  # index position in idx
        j = idx[i]  # retrieve index in z

        # retrieve range of l containing j
        l_range = np.arange(j - l + 1, j + l)

        assert (
            l_range[0] and l_range[-1] not in idx
        ), "Peaks selected within range of l."
        assert j == l_range[1], "Largest peak incorrectly indexed."
        assert Z[0][j] == np.max(Z), "Largest peak incorrectly identified."


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
        squared = fake_data**2
        summed = squared.sum()
        frob_norm = np.sqrt(summed)

        # this is how the function calculates Frobenius norm
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

    x = emg.preprocessing.flatten_signal(signal)
    x = emg.preprocessing.center_matrix(x)
    x = emg.preprocessing.extend_all_channels(x, 16)
    z = emg.preprocessing.whiten(x)

    n = 0

    z_peak_indices, z_peak_heights = emg.decomposition.initial_w_matrix(z)
    z_highest_peak = z_peak_indices[np.argmax(z_peak_heights)]

    # Initialize separation vectors and matrix B
    w_curr = z[:, z_highest_peak]
    w_prev = w_curr
    B = np.zeros((1088, 1))

    while linalg.norm(np.dot(w_curr.T, w_prev) - 1) > Tolx:

        w_prev = w_curr

        # Calculate separation vector
        b = np.dot(w_prev, z)
        g_b = emg.contrast.apply_contrast(b)
        A = (emg.contrast.apply_contrast(b, der=True)).mean()
        w_curr = (z * g_b).mean(axis=1) - A * w_prev

        # Orthogonalize and normalize separation vector
        w_curr = emg.decomposition.orthogonalize(w_curr, B)
        w_curr = emg.decomposition.normalize(w_curr)

        # If n exceeds max iteration exit while loop
        n += 1
        if n > max_iter:
            break

    assert (
        w_curr
        == emg.decomposition.separation(
            z,
            z[:, z_highest_peak],
            B,
            Tolx,
            emg.contrast.skew,
            emg.decomposition.gram_schmidt,
            max_iter,
        )
    ).all(), "Separation vector incorrectly calculated."


def test_deflate():
    """
    Run unit tests on deflate function from EMGdecomPy.
    """
    w_list = [np.array([1, 2, 3]), np.array([2, 4, 6])]

    B_list = [np.array([[2, 4, 6], [10, 20, 30]]), np.array([[1, 3, 4], [10, 20, 30]])]

    answers = [np.array([-1455, -2910, -4365]), np.array([-2836, -5710, -8546])]

    for i, answer in enumerate(answers):
        fx_answer = emg.decomposition.deflate(w_list[i], B_list[i])
        assert np.array_equal(fx_answer, answer), "Source deflated incorrectly."


def test_gram_schmidt():
    """
    Run unit tests on gram_schmidt function from EMGdecomPy.
    """
    # Arrays to multiply
    w_list = [np.array([7, 4, 6]), np.array([1, 1, 1])]

    # Dot product answers calculated by hand
    B_list = [
        np.array([[1, 1.2, 0], [2, -0.6, 0], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
    ]

    for i, w in enumerate(w_list):

        output = emg.decomposition.gram_schmidt(w, B_list[i])

        assert np.all(
            np.dot(B_list[i].T, output) == 0
        ), "Dot product of B.T and output not equal to 0"


def test_orthogonalize():
    """
    Run unit tests on orthogonalize from EMGdecomPy.
    """
    w = np.array([1, 5, 1])
    B = np.array([[20, 0, 0], [0, 345, 0], [0, 0, 0]])

    # Test the two orthogonalize options
    gram_s = emg.decomposition.orthogonalize(w, B, fun=emg.decomposition.gram_schmidt)
    d_flate = emg.decomposition.orthogonalize(w, B, fun=emg.decomposition.deflate)

    assert np.all(
        np.dot(B.T, gram_s) == 0
    ), "Gram-Schmidt orthoganlization incorrectly applied."
    assert np.all(d_flate != 0), "Incorrectly othogonalized."


def test_refinement():
    """
    Run unit test on refinement function from EMGdecomPy.
    """
    random_seed = 42
    gl_10 = loadmat("data/raw/GL_10.mat")
    signal = gl_10["SIG"]

    x = emg.preprocessing.flatten_signal(signal)
    x = emg.preprocessing.center_matrix(x)
    x = emg.preprocessing.extend_all_channels(x, 16)
    z = emg.preprocessing.whiten(x)
    B = np.zeros((1088, 1))
    Tolx = 10e-4
    max_iter = 10
    thresh = 20
    l = 31

    z_peak_indices, z_peak_heights = emg.decomposition.initial_w_matrix(z)
    z_highest_peak = z_peak_indices[np.argmax(z_peak_heights)]

    w_i = emg.decomposition.separation(
        z,
        z[:, z_highest_peak],
        B,
        Tolx,
        emg.contrast.skew,
        emg.decomposition.gram_schmidt,
        max_iter,
    )

    np.random.seed(random_seed)
    cv_prev = np.random.ranf()
    cv_curr = cv_prev * 0.9

    # Get output of refine function
    ref_output = emg.decomposition.refinement(
        w_i,
        z,
        i=0,
        l=l,
        sil_pnr=False,
        thresh=thresh,
        max_iter=max_iter,
        random_seed=random_seed,
        verbose=False,
    )

    for iter in range(max_iter):

        # a. Estimate the i-th source
        s_i = np.dot(w_i, z)  # w_i and w_i.T are equal as far as I know

        # Estimate pulse train pt_n with peak detection applied to the square of the source vector
        s_i2 = np.square(s_i)

        peak_indices, _ = find_peaks(s_i2, distance=l)

        # b. Use KMeans to separate large peaks from relatively small peaks, which are discarded
        kmeans = KMeans(n_clusters=2, random_state=random_seed)
        kmeans.fit(s_i2[peak_indices].reshape(-1, 1))
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

        # Create pulse train, where values are 0 except for when MU fires, which have values of 1
        pt_n = np.zeros_like(s_i2)
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

        if np.isnan(cv_curr):
            cv_curr = 0

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
    pnr_score = emg.decomposition.pnr(s_i2, peak_indices_a)

    sil = emg.decomposition.silhouette_score(s_i2, peak_indices_a)

    if pnr_score < thresh or cv_curr <= cv_prev or cv_curr == 0:
        test_output = np.zeros_like(w_i), np.zeros_like(s_i), np.array([]), 0, 0
    else:
        test_output = w_i, s_i, peak_indices_a, sil, pnr_score

    # Check the dimensions of the output: expect it to be same as input array
    assert (
        ref_output[0].shape == test_output[0].shape
    ), "Shape of refined array does not match shape of test array"

    assert np.allclose(
        ref_output[0], test_output[0]
    ), "Different separation vector from refinement and test functions"

    assert np.allclose(
        ref_output[1], test_output[1]
    ), "Different estimated source from refinement and test functions"

    assert np.allclose(
        ref_output[2], test_output[2]
    ), "Different firing times produced by refinement and test functions"

    assert (
        ref_output[3] == test_output[3]
    ), "Different SIL from refinement and test functions"

    assert (
        ref_output[4] == test_output[4]
    ), "Different PNR from refinement and test functions"


def test_pnr():
    """
    Run unit test on pnr function from EMGdecomPy.
    """
    samples = [3, 100, 1000]
    for i in samples:

        # Create data with peaks
        features, clusters = make_blobs(
            n_samples=i, n_features=1, centers=2, random_state=50
        )
        features = features.flatten() ** 2

        # Save indices of peaks
        peak_indices, _ = find_peaks(features)

        fx_pnr = emg.decomposition.pnr(features, peak_indices)

        # Calculate pnr by hand:
        pulse = []
        noise = []

        for j in features:

            condition = j in list(features[peak_indices])

            if condition == True:
                pulse.append(j)  # Numerator: peak height
            else:
                noise.append(j)  # Denominator: non-peak height

        # Calculate average height of each category
        pulse_avg = np.array(pulse).mean()
        noise_avg = np.array(noise).mean()

        # Log
        p_to_n = 10 * np.log10(pulse_avg / noise_avg)

        assert np.isclose(fx_pnr, p_to_n), "PNR incorrectly calculated."


def test_silhouette_score():
    """
    Run unit test on silhouette_score function from EMGdecomPy.
    """

    state_list = [50, 250]

    for i in state_list:
        # Create data with peaks
        features, clusters = make_blobs(
            n_samples=10, n_features=1, centers=3, random_state=i
        )
        features = features.flatten() * 100  # Flatten to find_peaks

        # Find indices of peaks
        peak_indices, _ = find_peaks(features)

        # Calculate SIL by hand
        pulse = []
        noise = []

        for j in features:
            condition = j in list(features[peak_indices])

            if condition == True:
                pulse.append(j)  # Coordinates of firing
            else:
                noise.append(j)  # Coordinates of noise

        # Calculate center of clusters
        pulse_center = np.mean(pulse)
        noise_center = np.mean(noise)

        intra = abs(pulse - pulse_center).sum() + abs(noise - noise_center).sum()
        inter = abs(pulse - noise_center).sum() + abs(noise - pulse_center).sum()

        numer = inter - intra
        denom = max(inter, intra)
        sil_by_hand = numer / denom

        sil = emg.decomposition.silhouette_score(features, peak_indices)

        assert np.isclose(
            sil, sil_by_hand
        ), "Inter and intra distances incorrectly calculated."
