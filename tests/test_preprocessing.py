# Copyright (C) 2022 Daniel King, Jasmine Ortega, Rada Rudyak, Rowan Sivanandam
# Test script for all functions defined in src/preprocessing.py

from emgdecompy import preprocessing as emg
import numpy as np
from scipy import linalg
from scipy.io import loadmat

# Test script for all functions defined in src/preprocessing.py


def create_emg_data(m=13, n=5, q=10):
    """
    Creates array (m, n) of arrays (1, q) with one empty subarray.

    Parameters
    ---------
    m : int
        Number of rows to set as outer array size. The default size is 13.
    n : int
        Number of entries for outer array to have in each row. The default size is 5.
    q : int
        Number of entries for inner array to have in each row. The default size is 10; the default shape is (1, 10).

    Returns
    -------
    raw : numpy.ndarray
    """
    # Intialize which subarray will be empty
    empty_row = np.random.randint(0, m)
    empty_col = np.random.randint(0, n)

    fake_data = np.zeros([m, n], dtype=object)

    # List of five sets of fake data
    for i in range(0, m):
        fake_data[i, :] = [np.random.randn(1, q)]  # same sequence for each row in array

    fake_data[empty_row, empty_col] = np.empty([0, 0])

    return fake_data


def test_extend_input_by_R():
    """
    Run unit tests on extend_input_by_R function from EMGdecomPy.
    """

    for i in range(0, 15):
        # Extension factor
        R = np.random.randint(1, 100)

        # Length of input array
        q = np.random.randint(1, 100)

        # Create input array
        x = np.random.rand(q)

        # Check input parameters
        assert R % 1 == 0, "Value of R must be an integer."
        assert R > 0, "Value of R must be greater than zero."
        assert (
            sum(x.shape) == x.shape[0]
        ), f"Input array must be one-dimensional eg. ({k},)"

        k = x.shape[0]

        testing = emg.extend_input_by_R(x, R)

        # Check values are properly extended
        assert (
            testing[1][0] == 0
        ), "Array not extended properly."  # First extended array
        assert testing.shape == (R + 1, x.shape[0]), "Shape of extended array incorrect"

        if R >= k:

            # If R >=k, last few arrays will be all zeroes
            assert testing[k - 1][-1] == x[0], "Array not extended properly."
            assert (
                np.count_nonzero(testing[k]) == 0
            ), f"Extended array should contain all zeros at testing[{k}]"
            assert (
                np.count_nonzero(testing[-1]) == 0
            ), "Extended array should contain all zeros in last row"

        else:
            assert testing[R][R] == x[0], "Array not extended not properly."
            assert (
                np.count_nonzero(testing[-1]) + R == k
            ), "Array not extended not properly."


def test_extend_all_channels():
    """
    Run unit tests on extend_input_all_channels function from EMGdecomPy.
    """
    for i in range(0, 15):

        # Initalize dimensions of test data
        x = np.random.randint(2, 100)
        y = np.random.randint(2, 100)
        z = np.random.randint(2, 1000)

        # Create test data + flatten
        fake = create_emg_data(x, y, z)
        flat = emg.flatten_signal(fake)

        # Input array must be two dimensional
        assert sum(flat.shape) > flat.shape[0], "Input array is not of shape M x K"

        m, k = flat.shape

        # Ensure that correct shape can be outputted
        assert m > 0, "Input array cannot be empty."
        assert k > 1, "Input array must contain more than one channel."

        # Negro, et al used R = 16
        R = np.random.randint(1, 30)

        # Test input parameters of extend_all_channels()
        assert R > 0, "Value of R must be greater than 0."
        assert R % 1 == 0, "Value of R must be an integer."

        # Extend channels
        ext = emg.extend_all_channels(flat, R)

        # Test output
        assert np.count_nonzero(ext[0]) == k, "Values extended incorrectly at ext[0]"
        assert ext.shape == (
            m * (R + 1),
            k,
        ), "Output array does not have shape M(R+1) x K"

        if (
            R > k
        ):  # If extension factor is bigger than length of array to extend, the last row is all zeros
            assert (
                np.count_nonzero(ext[-1]) == 0
            ), "Values incorrectly extended at ext[-1]"
        else:
            # Otherwise there should be R zeros in last row
            assert (
                np.count_nonzero(ext[-1]) + R == k
            ), "Values incorrectly extended at ext[-1]"


def test_flatten_signal():
    """
    Run unit tests on flatten_signal function from EMGdecomPy.
    """
    # Create fake data
    fake_data = []

    for i in range(0, 4):
        m = np.random.randint(2, 150)
        n = np.random.randint(2, 150)
        q = np.random.randint(2, 150)

        fake_data.append(create_emg_data(m, n, q))

    # Run tests on fake datasets
    for i in fake_data:

        # Test that input is correct
        assert type(i) == np.ndarray, "Input is not type numpy.ndarray"
        assert i.shape != (1, 1), "Input array is already one-dimensional."

        flat = emg.flatten_signal(i)

        # Shape of fake data
        m, n = i.shape
        q = flat.shape[1]

        # Test that inner arrays are correct length

        # If the first element is null array that was removed in flat,
        # Check the second element's shape for consistency
        if 0 not in i[0][0].shape:
            assert i[0][0].shape[1] == q, "Dimensions of inner array not the same."
        else:
            assert i[0][1].shape[1] == q, "Dimensions of inner array not the same."

        # Test that empty channel has been removed
        assert (m * n) != flat.shape[0], "Empty array not removed"
        
def test_butter_bandpass_filter():
    """
    Run unit test on butter_bandpass_filter function from EMGdecomPy.
    """
    
    gl_10 = loadmat("data/raw/GL_10.mat")
    raw = gl_10["SIG"]

    # select two channels from raw data
    data = raw[1, 1:3]
    
    d = emg.flatten_signal(data)
    x = emg.butter_bandpass_filter(d)
    
    assert type(x) == np.ndarray, "Incorrect datatype returned."
    assert x.shape == d.shape, "Incorrect shape returned."
    


def test_center_matrix():
    """
    Run unit tests on center_matrix function from EMGdecomPy.
    """
    x1 = np.array([[1, 2, 3], [4, 6, 8]])
    x2 = np.array([[[1, 2, 3], [4, 6, 8]], [[10, 13, 16], [17, 21, 25]]])

    # Assert center_matrix works on a 2D array
    assert (emg.center_matrix(x1)[0] == x1[0] - x1[0].mean()).all()
    assert (emg.center_matrix(x1)[1] == x1[1] - x1[1].mean()).all()

    # Assert center_matrix works on a 3D array
    assert (emg.center_matrix(x2)[0][0] == x2[0][0] - x2[0][0].mean()).all()
    assert (emg.center_matrix(x2)[0][1] == x2[0][1] - x2[0][1].mean()).all()
    assert (emg.center_matrix(x2)[1][0] == x2[1][0] - x2[1][0].mean()).all()
    assert (emg.center_matrix(x2)[1][1] == x2[1][1] - x2[1][1].mean()).all()


def test_whiten():
    """
    Run unit test on whitening function from EMGdecomPy.
    """
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    x_cent = emg.center_matrix(x)
    cov_mat = np.cov(x_cent, rowvar=True, bias=True)
    w, v = linalg.eig(cov_mat)
    reg_factor = w[round(len(w) / 2) :].mean()
    w = np.where(w < reg_factor, reg_factor, w)

    D = np.diag(w)
    D = np.sqrt(linalg.inv(D))
    D = D.real
    W = np.dot(np.dot(v, D), v.T)
    np.dot(W, x_cent)

    assert np.allclose(np.dot(W, x_cent), emg.whiten(x_cent))
