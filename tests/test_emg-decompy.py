import random 

R_one = 5
R_two = 10
x = np.array([1, 2, 3, 4, 5, 6, 7])

# Tests for extend_input_by_R

assert extend_input_by_R(x, R_one)[0][0] == x[0]
assert extend_input_by_R(x, R_one)[0][-1] == 0
assert extend_input_by_R(x, R_one).shape == (len(x), R_one + 1)
assert extend_input_by_R(x, R_one)[-1][0] == x[-1]
assert extend_input_by_R(x, R_one)[0][0] == extend_input_by_R(x, R_one)[1][1]

assert extend_input_by_R(x, R_two)[0][0] == x[0]
assert extend_input_by_R(x, R_two)[0][-1] == 0
assert extend_input_by_R(x, R_two).shape == (len(x), R_two + 1)
assert extend_input_by_R(x, R_two)[-1][0] == x[-1]
assert extend_input_by_R(x, R_two)[0][0] == extend_input_by_R(x, R_two)[1][1]

# Tests for extend_input_all_channels

x_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

assert extend_input_all_channels(x_mat, R_one).shape == (len(x_mat), len(x_mat[0]), R_one + 1)
assert extend_input_all_channels(x_mat, R_one)[0][0][0] == extend_input_all_channels(x_mat, R_one)[0][1][1]
assert extend_input_all_channels(x_mat, R_one)[0][0][-1] == 0
assert sum(extend_input_all_channels(x_mat, R_one)[-1][-1]) == sum(x_mat[-1])

assert extend_input_all_channels(x_mat, R_two).shape == (len(x_mat), len(x_mat[0]), R_two + 1)
assert extend_input_all_channels(x_mat, R_two)[0][0][0] == extend_input_all_channels(x_mat, R_two)[0][1][1]
assert extend_input_all_channels(x_mat, R_two)[0][0][-1] == 0
assert sum(extend_input_all_channels(x_mat, R_two)[-1][-1]) == sum(x_mat[-1])

# Tests for flatten_signal()

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
        for j in range(0, n):
            if i == empty_row and j == empty_col:
                fake_data[i][j] = np.array([])

            else:
                fake_data[i][j] = np.random.rand(1, q)
    return fake_data

def test_flatten_signal(data):
    """
    Run unit tests on flatten_signal function from emg-decomPy.

    Parameters
    ----------
    data : numpy.ndarray
        Array of arrays, with at least one empty sub-array.
    """
    # test that input is array 
    assert type(data) == np.ndarray, "Input is not type numpy.ndarray"

    # test that data has been flattened properly
    x, y = data.shape
    flat = flatten_signal(data)
    
    if data[1][0].shape[0] == 0: # unfortunate event that this value is the empty array
        assert np.allclose(data[1][1], flat[y]), "Flattened data values not aligned with original data" 
        
    else:
        assert np.allclose(data[1][0], flat[y]) or np.allclose(data[1][0], flat[y-1]), "Flattened data values not aligned with original data" 

    # test that empty channel has been removed 
    assert (x * y) != flatten_signal(data).shape[0], "Empty array not removed"
    
    
    # test fake arrays with single channel missing

    fake_data = create_emg_data() # default values
    test_flatten_signal(fake_data)

    fake_data = create_emg_data(m=2, n=3, q=150)
    test_flatten_signal(fake_data)

    fake_data = create_emg_data(m=50, n=30, q=100)
    test_flatten_signal(fake_data)

    fake_data = create_emg_data(m=15, n=25, q=10)
    test_flatten_signal(fake_data)
    

