from scipy.io import loadmat
import emgdecompy as emg
import numpy as np
import pandas as pd
import math
import pytest

@pytest.fixture
def mu():
    """
    Create fake indices of pulse trains for two motor units. 
    """
    # ptl 
    mu_values = np.array([[32, 90],[250, 300]])
    
    return mu_values

@pytest.fixture
def fx_data():
    """
    Create subset of EMG data to test with. 
    """
    # load data
    gl_10 = loadmat("data/raw/GL_10.mat")
    raw = gl_10["SIG"]
    
    # select two channels from raw data
    data = raw[1, 1:3]
    return data

def test_RMSE():
    """
    Run unit test on RMSE function from EMGdecomPy.
    """
    actual = np.array([0, 10, 50, 75])
    predicted = np.array([0.1, 10.1, 50.1, 75.1])
    
    # hand calculate MSE
    mse = np.sum((actual - predicted) ** 2) / len(actual)
    
    rmse = np.sqrt(mse)
    
    rmse_fx = emg.viz.RMSE(actual, predicted)
    
    assert np.isclose(rmse, rmse_fx), "RMSE was incorrectly calculated."


def test_muap_dict(fx_data, mu):
    """
    Run unit test on muap_dict function from EMGdecomPy.
    """

    # actual function
    fx = emg.viz.muap_dict(fx_data, mu, l=2)

    # hand calcuating avg
    l = 2

    raw_flat = emg.preprocessing.flatten_signal(fx_data) 

    mu = mu.squeeze() 

    all_peak_idx = [] # list of all peaks in pulse train  

    for i in mu:
        k = 0 

        while k <= 1:
            firing = i[k]
            
            if np.less(firing, l) == True:
                idx = np.arange(firing- l, firing + l + 1) 
                neg_idx = abs(firing - l)
                idx[:neg_idx] = np.repeat(0, neg_idx)
            
            else:
                idx = np.arange(firing - l, firing + l + 1) 
                
            all_peak_idx.append(idx)
            k += 1 

    peaks = raw_flat[:, all_peak_idx]

    signal = np.zeros((2, 2, 5))

    n_mu = mu.shape[1] 
    
    for i in range(0, n_mu):
        if i == 0:
            avg = peaks[:, 0:n_mu].mean(axis=1)
        else: 
            avg = peaks[:, n_mu:].mean(axis=1)
        signal[i] = avg
    
    # test sample length (plotting purposes)
    x, y, z = signal.shape

    assert y * z == len(fx["mu_0"]["signal"]), "Signal length incorrect."
    assert y * z == len(fx["mu_1"]["signal"]), "Signal length incorrect."
    
    assert y * z == len(fx["mu_0"]["sample"]), "Sample length incorrect."
    assert y * z == len(fx["mu_1"]["sample"]), "Sample length incorrect."
        
    assert y * z == len(fx["mu_0"]["channel"]), "Channel length incorrect."
    assert y * z == len(fx["mu_1"]["channel"]), "Channel length incorrect."
    
    # test values of avg signal
    assert np.array_equal(signal[0].flatten(), fx["mu_0"]["signal"]), "Average of motor unit signal incorrectly calculated."
    assert np.array_equal(signal[1].flatten(), fx["mu_1"]["signal"]), "Average of motor unit signal incorrectly calculated."

    
def test_muap_plot(fx_data, mu):
    """
    Run unit test on muap_plot function from EMGdecomPy.
    """

    shape_dict = emg.viz.muap_dict(fx_data, mu, l=2)
    
    for i in range(0, 2): # test motor unit 1 and 2
        plots = emg.viz.muap_plot(shape_dict, i)

        # test dictionary correctly converted to df
        len_data = len(plots.data)
        len_input = len(shape_dict[f"mu_{i}"]["sample"])

        assert len_data == len_input, "Input data incorrectly converted to df"
        assert plots.encoding.x.shorthand == "sample", "Incorrect data on x-axis"
        assert plots.encoding.y.shorthand == "signal", "Incorrect data on y-axis"

