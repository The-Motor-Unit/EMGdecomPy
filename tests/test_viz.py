# Copyright (C) 2022 Daniel King, Jasmine Ortega, Rada Rudyak, Rowan Sivanandam
# Test script for all functions defined in src/viz.py

from scipy.io import loadmat
import emgdecompy as emg
import numpy as np
import altair as alt
import panel
import pytest

# Note: Fixtures are special PyTest objects calld into individual tests,
# they are useful when data is repeatedly required to test functions


@pytest.fixture
def mu():
    """
    Create fake indices of pulse trains for two motor units.
    """
    # Pulse train
    mu_values = np.array([[32, 90], [250, 300]])

    return mu_values


@pytest.fixture
def fx_data():
    """
    Create subset of EMG data to test with.
    """
    # Load data
    gl_10 = loadmat("data/raw/GL_10.mat")
    raw = gl_10["SIG"]

    # Select two channels from raw data
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


@pytest.fixture
def fake_decomp():
    """
    Create small decomposed dictionary to test with.
    """
    decompose = {
        "B": np.array([[-7.53, 1.50, 8.62, -5.42], [-7, 1, 8, -5]]),
        "MUPulses": np.array(
            [[10, 25, 35, 45, 60, 85], [100, 250, 350, 450, 600, 850]]
        ),
        "SIL": [0.95, 0.95],
        "PNR": [21.00, 21.00],
    }
    return decompose


def test_RMSE():
    """
    Run unit test on RMSE function from EMGdecomPy.
    """
    # Create data
    actual = np.array([0, 10, 50, 75])
    predicted = np.array([0.1, 10.1, 50.1, 75.1])

    # Hand calculate mean squared error
    mse = np.sum((actual - predicted) ** 2) / len(actual)

    # Hand calculate root mean squared error
    rmse = np.sqrt(mse)

    rmse_fx = emg.viz.RMSE(actual, predicted)

    assert np.isclose(rmse, rmse_fx), "RMSE was incorrectly calculated."


def test_muap_dict(fx_data, mu):
    """
    Run unit test on muap_dict function from EMGdecomPy.
    """

    # Function to test
    fx = emg.viz.muap_dict(fx_data, mu, l=2)

    # Create muap_dict using a different method
    raw_flat = emg.preprocessing.flatten_signal(fx_data)
    l = 2
    mu = mu.squeeze()

    all_peak_idx = []  # List of all peaks in pulse train

    # For each motor unit, collect the indices around a firing (+/-l)
    # This allows us to visualize the entire shape of the peak
    for i in mu:
        k = 0

        while k <= 1:
            firing = i[k]

            # Edge case where MU fires at value < l, prevents negative indexing
            if np.less(firing, l) == True:
                idx = np.arange(firing - l, firing + l + 1)
                neg_idx = abs(firing - l)
                idx[:neg_idx] = np.repeat(0, neg_idx)

            else:
                idx = np.arange(firing - l, firing + l + 1)

            all_peak_idx.append(idx)
            k += 1

    # Grab values of peaks + surrouding range (+/- l)
    peaks = raw_flat[:, all_peak_idx]

    signal = np.zeros((2, 2, 5))

    n_mu = mu.shape[1]

    # Calculate average shape of peaks across a single channel
    for i in range(0, n_mu):
        if i == 0:
            avg = peaks[:, 0:n_mu].mean(axis=1)
        else:
            avg = peaks[:, n_mu:].mean(axis=1)
        signal[i] = avg

    # Test sample length (plotting purposes)
    x, y, z = signal.shape

    assert y * z == len(fx["mu_0"]["signal"]), "Signal length incorrect."
    assert y * z == len(fx["mu_1"]["signal"]), "Signal length incorrect."

    assert y * z == len(fx["mu_0"]["sample"]), "Sample length incorrect."
    assert y * z == len(fx["mu_1"]["sample"]), "Sample length incorrect."

    assert y * z == len(fx["mu_0"]["channel"]), "Channel length incorrect."
    assert y * z == len(fx["mu_1"]["channel"]), "Channel length incorrect."

    # Test values of avg signal
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
    # Create dictionary to test
    peak_dict = emg.viz.muap_dict_by_peak(fx_data, 100, mu_index=1, l=2)
    channel = peak_dict["mu_1"]["channel"]

    # l = 1/2 length of firing
    l = 2
    x = fx_data.shape[0]
    rng = l * 2 + 1  # entire range of firing

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

    for i in range(0, 2):  # Test motor unit 1 and 2
        plots = emg.viz.muap_plot(shape_dict, i)

        # Test dictionary correctly converted to df
        len_data = len(plots.data)
        len_input = len(shape_dict[f"mu_{i}"]["sample"])

        assert len_data == len_input, "Input data incorrectly converted to df"
        assert plots.encoding.x.shorthand == "sample", "Incorrect data on x-axis"
        assert plots.encoding.y.shorthand == "signal", "Incorrect data on y-axis"


def test_mismatch_scores(avg_mu_shape, avg_peak_shape):
    """
    Run unit test on mismatch function from EMGdecomPy.
    """

    # Test error across all channels for mu_1
    fx_output = emg.viz.mismatch_score(avg_mu_shape, avg_peak_shape, mu_index=1)

    x = avg_mu_shape["mu_1"]["signal"]
    y = avg_peak_shape["mu_1"]["signal"]

    hand_RMSE = np.sqrt(np.sum((x - y) ** 2) / len(x))

    assert hand_RMSE == fx_output, "RMSE incorrectly calculated."

    for num, i in enumerate(avg_peak_shape):

        # Test error across a single channel for both motor units
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
    # Test standard orientation

    std = emg.viz.channel_preset(preset="standard")

    assert std["sort_order"][-1] == 63, "Standard orientation incorrect."
    assert len(std["sort_order"]) == 64, "Standard orientation incorrect."
    assert std["cols"] == 8, "Standard orientation incorrect."

    # Test vert63 orientation

    std = emg.viz.channel_preset(preset="vert63")

    assert std["sort_order"][-1] == 24, "Standard orientation incorrect."
    assert std["sort_order"][0] == 63, "Standard orientation incorrect."
    assert len(std["sort_order"]) == 64, "Standard orientation incorrect."
    assert std["cols"] == 5, "Standard orientation incorrect."


def test_pulse_plot(fx_data):
    """
    Run unit test on pulse_plot function from EMGdecomPy.
    """
    # Note: the individual plots are not accessible in a concat'd dashboard,
    # so these tests are rather simple

    # Create two motor unit pulse trains
    pt = np.array([[10, 60, 120], [15, 65, 125]])

    # Pre-process data
    signal = emg.preprocessing.flatten_signal(fx_data)
    signal = np.apply_along_axis(
        emg.preprocessing.butter_bandpass_filter,
        axis=1,
        arr=signal,
        lowcut=10,
        highcut=900,
        fs=2048,
        order=6,
    )

    # Calculate square mean of centered data
    centered = emg.preprocessing.center_matrix(signal)
    c_sq = centered**2
    c_sq_mean = c_sq.mean(axis=0)
    c_sq_mean

    for i, j in enumerate(pt):
        plt = emg.viz.pulse_plot(pt, c_sq_mean, mu_index=i)

        # Access dataframe used in plot
        df = plt.data
        df = df["Pulse"].to_numpy()

        assert np.all(df == j), "Incorrect data in plot df."

    df_cols = ["Pulse", "Strength", "Motor Unit", "Hz", "seconds"]

    assert np.all(plt.data.columns == df_cols), "Incorrect data in df."


def test_select_peak(fx_data, mu):
    """
    Run unit test on select_peak function from EMGdecomPy.
    """
    dic = emg.viz.muap_dict(fx_data, mu, l=2)

    # Test empty peak selection
    select = []
    pulse = [[100], [200]]

    # Altair objects, like plot, can be indexed into to access individual plots
    plot = emg.viz.select_peak(
        selection=select, mu_index=1, raw=fx_data, shape_dict=dic, pt=pulse
    )

    assert len(plot[0][0].object.data) == 10, "Incorrect data plotted."
    assert (
        plot[0][0].object.encoding.facet.shorthand == "channel"
    ), "Plots incorrectly facetted."
    assert plot[0][0].object.encoding.facet.columns == 8, "Plots incorrectly facetted."
    assert (
        plot[0][0].object.encoding.x.shorthand == "sample"
    ), "Incorrect x-axis plotted."
    assert (
        plot[0][0].object.encoding.y.shorthand == "signal"
    ), "Incorrect y-axis plotted."


def test_dashboard(fake_decomp, fx_data):
    """
    Run unit test on dashboard function from EMGdecomPy.
    """
    # There are not a lot of attributes to test for with Panel objects
    for i, decomp_pulse in enumerate(fake_decomp["MUPulses"]):

        dash = emg.viz.dashboard(fake_decomp, fx_data, i)
        df = dash[0].object.data
        df_pulses = df.Pulse

        assert (
            type(dash[0].object) == alt.vegalite.v4.api.VConcatChart
        ), "Object returned is not concatenated plots."

        # Check that plotted pulses match input
        assert np.all(df_pulses == decomp_pulse), "MU Pulses incorrectly plotted."


def test_visualize_decomp(fake_decomp, fx_data):
    """
    Run unit test on visualize_decomp function from EMGdecomPy.
    """

    x = emg.viz.visualize_decomp(fake_decomp, fx_data)

    # Concat'd Panel objects can be indexed into
    # x[0] are the widgets (dropdown menus)
    # x[1] are the actual plots
    assert x[0][0].values == [0, 1]
    assert x[0][1].values == ["standard", "vert63"]
    assert x[0][2].values == ["RMSE"]
    assert type(x[1][0]) == panel.layout.base.Column
