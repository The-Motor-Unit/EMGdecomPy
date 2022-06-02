from scipy.io import loadmat
import emgdecompy as emg
import numpy as np
import pandas as pd
import math

def test_muap_dict():
    """
    Run unit test on muap_dict function from EMGdecomPy.
    """
    # load data
    gl_10 = loadmat("data/raw/GL_10.mat")
    raw = gl_10["SIG"]

    # ptl
    mu = np.array([[32, 90],[250, 300]])

    # select two channels from raw data
    fx_data = raw[1, 1:3] 

    # actual function 
    fx = emg.viz.muap_dict(fx_data, mu, l=2)

    # hand calcuating avg
    l = 2

    raw_flat = emg.preprocessing.flatten_signal(fx_data) 

    mu = mu.squeeze() 

    all_peak_idx = [] # list of all peaks in pulse train  
    indx = [] # temporary index of pulse train

    for i in mu:
        k = 0 

        while k <= 1:
            firing = i[k]
            idx = np.arange(firing - l, firing + l) # need to add +1 to this once fx fixed
            all_peak_idx.append(idx)
            k += 1 

    peaks = raw_flat[:, all_peak_idx]

    signal = np.zeros((2, 2, 4))

    n_mu = mu.shape[1] 

    for i in range(0, n_mu):
        if i == 0:
            avg = peaks[:, 0:n_mu].mean(axis=1)
        else: 
            avg = peaks[:, n_mu:].mean(axis=1)
        signal[i] = avg

    # test sample length (plotting purposes)
    x, y, z = signal.shape
    assert y * z == len(fx["mu_0"]["sample"]), "Signal length incorrect."
    assert y * z == len(fx["mu_1"]["sample"]), "Signal length incorrect."

    # test values of avg signal
    assert np.array_equal(signal[0].flatten(), fx["mu_0"]["signal"]), "Average of motor unit signal incorrectly caluclated."
    assert np.array_equal(signal[1].flatten(), fx["mu_1"]["signal"]), "Average of motor unit signal incorrectly caluclated."

    
def test_muap_plot():
    """
    Run unit test on muap_plot function from EMGdecomPy.
    """
    # load data + subset two channels
    gl_10 = loadmat("data/raw/GL_10.mat")
    raw = gl_10["SIG"]
    fx_data = raw[1, 1:3]

    # ptl 
    mu = np.array([[32, 90],[250, 300]])

    shape_dict = emg.viz.muap_dict(fx_data, mu, l=2)
    
    for i in range(0, 2): # test motor unit 1 and 2
        plots = emg.viz.muap_plot(shape_dict, i)

        # test dictionary correctly converted to df
        len_data = len(plots.data)
        len_input = len(shape_dict[f"mu_{i}"]["sample"])

        assert len_data == len_input, "Input data incorrectly converted to df"
        assert plots.encoding.x.shorthand == "sample", "Incorrect data on x-axis"
        assert plots.encoding.y.shorthand == "signal", "Incorrect data on y-axis"

        # test number of pages needed to display all channels
        channels = fx_data.shape[0]
        pages = math.ceil(channels/12)

        # pages in plot
        page_len = int(plots.encoding.facet.title[-1])

        assert page_len == pages, "Incorrect number of pages displayed."