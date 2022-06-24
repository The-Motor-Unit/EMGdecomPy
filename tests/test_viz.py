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
    mu_values = np.array([[32, 90], [250, 300]])

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


@pytest.fixture
def avg_mu_shape():
    """
    Create simple MU shape data for testing.
    """

    avg_mu_shapes = {
        "mu_0": {
            "signal": np.array([100, 115, 125, 117, 110, 100, 115, 125, 117, 110]),
            "channel": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        },
        "mu_1": {
            "signal": np.array([10, 15, 25, 17, 10, 10, 15, 25, 17, 10]),
            "channel": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        },
    }
    return avg_mu_shapes


@pytest.fixture
def avg_peak_shape():
    """
    Create simple MU shape data for testing.
    """
    avg_peak_shapes = {
        "mu_0": {
            "signal": np.array([10, 10, 10, 10, 10, 20, 20, 20, 20, 20]),
            "channel": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        },
        "mu_1": {
            "signal": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "channel": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        },
    }
    return avg_peak_shapes


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

    # hand calculating avg
    l = 2

    raw_flat = emg.preprocessing.flatten_signal(fx_data)

    mu = mu.squeeze()

    all_peak_idx = []  # list of all peaks in pulse train

    for i in mu:
        k = 0

        while k <= 1:
            firing = i[k]

            if np.less(firing, l) == True:
                idx = np.arange(firing - l, firing + l + 1)
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
    assert np.array_equal(
        signal[0].flatten(), fx["mu_0"]["signal"]
    ), "Average of motor unit signal incorrectly calculated."
    assert np.array_equal(
        signal[1].flatten(), fx["mu_1"]["signal"]
    ), "Average of motor unit signal incorrectly calculated."


def test_muap_dict_by_peak(fx_data):
    """
    Run unit test on muap_dict_by_peak function from EMGdecomPy.
    """
    # create dictionary to test
    peak_dict = emg.viz.muap_dict_by_peak(fx_data, 100, mu_index=1, l=2)
    channel = peak_dict["mu_1"]["channel"]

    l = 2
    x = fx_data.shape[0]
    rng = l * 2 + 1

    sample_range = np.arange(0, rng)
    signal = peak_dict["mu_1"]["signal"]

    assert type(signal[0]) == np.float64, "Signal dictionary is empty."
    assert len(channel) == x * rng, "Length of channel data incorrect."
    assert np.all(
        sample_range == peak_dict["mu_1"]["sample"][0:rng]
    ), "Range centered around firing peak is incorrect."


def test_muap_plot(fx_data, mu):
    """
    Run unit test on muap_plot function from EMGdecomPy.
    """

    shape_dict = emg.viz.muap_dict(fx_data, mu, l=2)

    for i in range(0, 2):  # test motor unit 1 and 2
        plots = emg.viz.muap_plot(shape_dict, i)

        # test dictionary correctly converted to df
        len_data = len(plots.data)
        len_input = len(shape_dict[f"mu_{i}"]["sample"])

        assert len_data == len_input, "Input data incorrectly converted to df"
        assert plots.encoding.x.shorthand == "sample", "Incorrect data on x-axis"
        assert plots.encoding.y.shorthand == "signal", "Incorrect data on y-axis"


def test_mismatch_scores(avg_mu_shape, avg_peak_shape):
    """
    Run unit test on mismatch function from EMGdecomPy.
    """

    # test error across all channels for mu_1
    fx_output = emg.viz.mismatch_score(avg_mu_shape, avg_peak_shape, mu_index=1)

    x = avg_mu_shape["mu_1"]["signal"]
    y = avg_peak_shape["mu_1"]["signal"]

    hand_RMSE = np.sqrt(np.sum((x - y) ** 2) / len(x))

    assert hand_RMSE == fx_output, "RMSE incorrectly calculated."

    for num, i in enumerate(avg_peak_shape):

        # test single channel error for single motor unit
        fx_output = emg.viz.mismatch_score(
            avg_mu_shape, avg_peak_shape, mu_index=num, channel=0
        )

        x = avg_mu_shape[i]["signal"][:5]
        y = avg_peak_shape[i]["signal"][:5]

        hand_RMSE = np.sqrt(np.sum((x - y) ** 2) / len(x))

        assert np.isclose(hand_RMSE, fx_output), "RMSE incorrectly calculated."


def test_channel_preset():
    """
    Run unit test on channel_preset function from EMGdecomPy.
    """
    # test standard orientation

    std = emg.viz.channel_preset(name="standard")

    assert std["sort_order"][-1] == 63, "Standard orientation incorrect."
    assert len(std["sort_order"]) == 64, "Standard orientation incorrect."
    assert std["cols"] == 8, "Standard orientation incorrect."

    # test vert63 orientation

    std = emg.viz.channel_preset(name="vert63")

    assert std["sort_order"][-1] == 24, "Standard orientation incorrect."
    assert std["sort_order"][0] == 63, "Standard orientation incorrect."
    assert len(std["sort_order"]) == 64, "Standard orientation incorrect."
    assert std["cols"] == 5, "Standard orientation incorrect."
