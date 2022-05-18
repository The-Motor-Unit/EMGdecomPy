from emgdecompy import emgdecompy as emg
import random 
import numpy as np
from scipy import linalg

def test_extend_input_by_R():
    """
    Run unit tests on extend_input_by_R function from EMGdecomPy.
    """

    for i in range(0, 15): 
        R = np.random.randint(1, 100) # to extend 
        
        assert R % 1 == 0, "Value of R is not an integer."  
        assert R > 0 , "Value of R must be greater than zero."

        # length of input array 
        if R == 1:
            q = 1 

        else: 
            q = np.random.randint(1, R)

        middle = round(q/2)    
        x = np.random.rand(q) # create input array

        testing = emg.extend_input_by_R(x, R)

        assert testing[1][0] == 0, "Array not extended properly." # check first value 
        assert testing[middle][middle] == x[0] # check middle value 
        assert testing[q-1][q-1] == x[0], "Array not extended properly." # check end value 
        assert testing.shape == (R+1, x.shape[0]), "Shape of extended array incorrect"


def test_extend_all_channels():
    """
    Run unit tests on extend_input_all_channels function from EMGdecomPy.
    """
    for i in range(0, 15):
    
        # initalize dimensions of test data 
        x = np.random.randint(2, 100)
        y = np.random.randint(2, 100)
        z = np.random.randint(2, 1000) 

        # create test data + flatten
        fake = emg.create_emg_data(x, y, z)
        flat = emg.flatten_signal(fake)

        # input array must be two dimensional 
        assert sum(flat.shape) > flat.shape[0], "Input array is not of shape M x K"

        m, k = flat.shape

        # ensure that correct shape can be outputted
        assert m > 0, "Input array cannot be empty."
        assert k > 1, "Input array must contain more than one channel." 

        # Negro, et al used R = 16
        R = np.random.randint(1, 30)

        # test input parameters of extend_all_channels()
        assert R > 0, "Value of R must be greater than 0."
        assert R % 1 == 0, "Value of R must be an integer."

        # extend channels
        ext = emg.extend_all_channels(flat, R)

        # test output 
        assert np.count_nonzero(ext[0]) == k, "Values extended incorrectly at ext[0]"
        assert ext.shape == (m * (R+1), k), "Output array does not have shape M(R+1) x K"

        if R > k: # if extension factor is bigger than length of array to extend, the last row is all zeros
            assert np.count_nonzero(ext[-1]) == 0, "Values incorrectly extended at ext[-1]" 
        else:
            # otherwise there should be R zeros in last row
            assert np.count_nonzero(ext[-1]) + R == k, "Values incorrectly extended at ext[-1]" 
            
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

def test_whiten():
    """
    Run unit test on whitening function from EMGdecomPy.
    """
    x = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
    x_cent = emg.center_matrix(x)
    cov_mat = np.cov(x_cent, rowvar=True, bias=True)
    w, v = linalg.eig(cov_mat)
    #w += w[:len(w) / 2].mean()
    D = np.diag(w)
    D = np.sqrt(linalg.inv(D))
    D = D.real.round(4)
    W = np.dot(np.dot(v, D), v.T)
    np.dot(W, x_cent)

    assert np.allclose(np.dot(W, x_cent), emg.whiten(x))