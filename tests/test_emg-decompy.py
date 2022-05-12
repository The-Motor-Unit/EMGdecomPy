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

def test_flatten_signal():
    """
    Run unit tests on flatten_signal function from emg-decomPy

    Parameters
    ----------
    data : numpy.ndarray
        Array of arrays, with at least one empty sub-array.
    """
    fake_data = []

    for i in range(0,4):
        m = np.random.randint(1, 150)
        n = np.random.randint(1, 150)
        q = np.random.randint(1, 150)

        fake_data.append(create_emg_data(m, n, q))
        
    for i in fake_data:
    
        # test that input is array 
        assert type(i) == np.ndarray, "Input is not type numpy.ndarray"

        # test that data has been flattened properly
        x, y = i.shape
        flat = flatten_signal(i)

        if i[1][0].shape[0] == 0: # if first value in second row is empty array 
            assert np.allclose(i[1][1], flat[y]), "Flattened data values not aligned with original data" 

        else:
            assert np.allclose(i[1][0], flat[y]) or np.allclose(i[1][0], flat[y-1]), "Flattened data values not aligned with original data" 

        # test that empty channel has been removed 
        assert (x * y) != flatten_signal(i).shape[0], "Empty array not removed"