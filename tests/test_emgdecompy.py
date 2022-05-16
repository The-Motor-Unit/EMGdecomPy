from emgdecompy import emgdecompy as emg
import random 
import numpy as np

def test_extend_input_by_R():
    """
    Run unit tests on extend_input_by_R function from EMGdecomPy.
    """
    R_one = 5
    R_two = 10
    x = np.array([1, 2, 3, 4, 5, 6, 7])

    assert emg.extend_input_by_R(x, R_one)[0][0] == x[0]
    assert emg.extend_input_by_R(x, R_one)[0][-1] == 0
    assert emg.extend_input_by_R(x, R_one).shape == (len(x), R_one + 1)
    assert emg.extend_input_by_R(x, R_one)[-1][0] == x[-1]
    assert emg.extend_input_by_R(x, R_one)[0][0] == emg.extend_input_by_R(x, R_one)[1][1]

    assert emg.extend_input_by_R(x, R_two)[0][0] == x[0]
    assert emg.extend_input_by_R(x, R_two)[0][-1] == 0
    assert emg.extend_input_by_R(x, R_two).shape == (len(x), R_two + 1)
    assert emg.extend_input_by_R(x, R_two)[-1][0] == x[-1]
    assert emg.extend_input_by_R(x, R_two)[0][0] == emg.extend_input_by_R(x, R_two)[1][1]

def test_extend_input_all_channels():
    """
    Run unit tests on extend_input_all_channels function from EMGdecomPy.
    """
    R_one = 5
    R_two = 10
    x_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert emg.extend_input_all_channels(x_mat, R_one).shape == (len(x_mat), len(x_mat[0]), R_one + 1)
    assert emg.extend_input_all_channels(x_mat, R_one)[0][0][0] == emg.extend_input_all_channels(x_mat, R_one)[0][1][1]
    assert emg.extend_input_all_channels(x_mat, R_one)[0][0][-1] == 0
    assert sum(emg.extend_input_all_channels(x_mat, R_one)[-1][-1]) == sum(x_mat[-1])

    assert emg.extend_input_all_channels(x_mat, R_two).shape == (len(x_mat), len(x_mat[0]), R_two + 1)
    assert emg.extend_input_all_channels(x_mat, R_two)[0][0][0] == emg.extend_input_all_channels(x_mat, R_two)[0][1][1]
    assert emg.extend_input_all_channels(x_mat, R_two)[0][0][-1] == 0
    assert sum(emg.extend_input_all_channels(x_mat, R_two)[-1][-1]) == sum(x_mat[-1])

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
    # intialize which subarray will be empty
    empty_row = np.random.randint(0, m)
    empty_col = np.random.randint(0, n)

    fake_data = np.zeros([m, n], dtype=object)

    for i in range(0, m):
        fake_data[i, :] = [np.random.randn(1,q)] # same sequence for each row in array

    fake_data[empty_row, empty_col] = np.array([])
    
    return fake_data

def test_flatten_signal():
    """
    Run unit tests on flatten_signal function from EMGdecomPy.
    """
    # create fake data 
    fake_data = []

    for i in range(0,4):
        m = np.random.randint(1, 150)
        n = np.random.randint(1, 150)
        q = np.random.randint(1, 150)

        fake_data.append(create_emg_data(m, n, q))
           

    # run tests on fake datasets
    for i in fake_data:
    
        # test that input is correct
        assert type(i) == np.ndarray, "Input is not type numpy.ndarray"
        assert i.shape != (1,1), "Input array is already one-dimensional."

        flat = emg.flatten_signal(i)
        
        # shape of fake data 
        m, n = i.shape
        q = flat.shape[1]
        
        # test that inner arrays are correct length 
        assert i[0][0].shape[1] == q, "Dimensions of inner array not the same."
        
        # test that empty channel has been removed 
        assert (m * n) != flat.shape[0], "Empty array not removed"

def test_center_matrix():
    """
    Run unit tests on center_matrix function from EMGdecomPy.
    """
    x1 = np.array([[1, 2, 3], [4, 6, 8]])
    x2 = np.array([[[1, 2, 3], [4, 6, 8]], [[10, 13, 16], [17, 21, 25]]])
    
    # assert center_matrix works on a 2D array
    assert (emg.center_matrix(x1)[0] == x1[0] - x1[0].mean()).all()
    assert (emg.center_matrix(x1)[1] == x1[1] - x1[1].mean()).all()
    
    # assert center_matrix works on a 3D array
    assert (emg.center_matrix(x2)[0][0] == x2[0][0] - x2[0][0].mean()).all()
    assert (emg.center_matrix(x2)[0][1] == x2[0][1] - x2[0][1].mean()).all()
    assert (emg.center_matrix(x2)[1][0] == x2[1][0] - x2[1][0].mean()).all()
    assert (emg.center_matrix(x2)[1][1] == x2[1][1] - x2[1][1].mean()).all()

