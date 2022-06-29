# Copyright (C) 2022 Daniel King, Jasmine Ortega, Rada Rudyak, Rowan Sivanandam
# This script contains functions used to visualize the results of the
# blind source separation algorithm based off of Francesco Negro et al 2016 J. Neural Eng. 13 026027.

from codecs import raw_unicode_escape_decode
import numpy as np
import pandas as pd
import altair as alt
import panel as pn
from panel.interact import interact, fixed
import math
from sklearn.metrics import mean_squared_error
from emgdecompy.preprocessing import (
    flatten_signal,
    center_matrix,
    butter_bandpass_filter,
)

pn.extension("vega")


def RMSE(arr1, arr2):
    """
    Evaluates root mean square error for two series.

    Parameters
    ----------
        arr1: array
            First series.
        arr2: array
            Second series.

    Returns
    -------
        float
            Root mean square error of arr1 vs arr2.

    Examples
    --------
        >>> arr1 = [3, 4, 4, 9, 12]
        >>> arr2 = [3, 5, 3, 9, 11]
        >>> RMSE(arr1, arr2)
        0.7745966692414834

    """
    MSE = mean_squared_error(arr1, arr2)
    RMSE = math.sqrt(MSE)

    return RMSE


def mismatch_score(mu_data, peak_data, mu_index, method="RMSE", channel=-1):
    """
    Evaluates how well a given peak contributes to a given MUAP.
    This is called by the muap_plot() function and is used to
    include error in the title of the muap plot.

    Parameters
    ----------
        mu_data: dict
            Dictionary containing MUAP shapes for each motor unit.
        peak_data: dict
            Dictionary containing shapes for a given peak per channel.
        mu_index: int
            Index of motor unit to examine.
        method: str
            Metric to use for evaluating discrepency between mu_data and peak_data.
            Default: "RMSE"
        channel: int
            Channel to run evaluation on.
            Default: -1 which means average of all channels.

    Returns
    -------
        float
            Metric calculating difference between MU data vs Peak data.
            Default: Root Mean Square Error of MU data vs Peak data.
    """
    score = 0

    if channel == -1:  # For all channels, we can just
        # straight up compare RMSE across the board
        mu_sig = mu_data[f"mu_{mu_index}"]["signal"]
        peak_sig = peak_data[f"mu_{mu_index}"]["signal"]
        if method == "RMSE":
            score = RMSE(mu_sig, peak_sig)

    else:  # Otherwise, filter for a given channel
        # filter mu_data for signal data that channel
        indexes = np.where(mu_data[f"mu_{mu_index}"]["channel"] == channel)
        mu_sig = mu_data[f"mu_{mu_index}"]["signal"][indexes]

        indexes = np.where(peak_data[f"mu_{mu_index}"]["channel"] == channel)
        peak_sig = peak_data[f"mu_{mu_index}"]["signal"][indexes]
        if method == "RMSE":
            score = RMSE(mu_sig, peak_sig)

    return score


def muap_dict(raw, pt, l=31):
    """
    Returns multi-level dictionary containing sample number, average signal, and channel
    for each motor unit by averaging the peak shapes over every firing for each MUAP.

    Parameters
    ----------
        raw: numpy.ndarray
            Raw EMG signal.
        pt: numpy.ndarray
            Multi-dimensional array containing indices of firing times
            for each motor unit.
        l: int
            One half of the action potential discharge time in samples.
            Default value of 31 corresponds to approximately 15 ms at a
            sampling rate of 2048 Hz.

    Returns
    -------
        dict
            Dictionary containing MUAP shapes for each motor unit.
    """
    raw = flatten_signal(raw)
    channels = raw.shape[0]
    shape_dict = {}
    pt = pt.squeeze()  # Remove 1-dimensional axis for clean looping

    for i in range(pt.shape[0]):
        pt[i] = pt[i].squeeze()

        # Create array to contain indices of peak shapes
        ptl = np.zeros((pt[i].shape[0], l * 2 + 1), dtype="int")

        for j, k in enumerate(pt[i]):
            ptl[j] = np.arange(k - l, k + l + 1)

            # This is to ensure that if early peak happens before half of AP discharge time.
            # that there is no negative indices
            if k < l:
                ptl[j] = np.arange(k - l, k + l + 1)
                neg_idx = abs(k - l)
                ptl[j][:neg_idx] = np.repeat(0, neg_idx)

        ptl = ptl.flatten()

        # Create channel index of each peak
        channel_index = np.repeat(np.arange(channels), l * 2 + 1)

        # Get sample number of each position along each peak
        sample = np.arange(l * 2 + 1)
        sample = np.tile(sample, channels)

        # Get average signals from each channel
        signal = (
            raw[:, ptl]
            .reshape(channels, ptl.shape[0] // (l * 2 + 1), l * 2 + 1)
            .mean(axis=1)
            .flatten()
        )

        shape_dict[f"mu_{i}"] = {
            "sample": sample,
            "signal": signal,
            "channel": channel_index,
        }

    return shape_dict


def muap_dict_by_peak(raw, peak, mu_index=0, l=31):
    """
    Returns the dictionary of shapes for a selected peak, by channel.
    It is called by the select_peak() function when a peak is selected by a user.

    Parameters
    ----------
        raw: numpy.ndarray
            Raw EMG signal.
        peak: int
            Peak timing to plot.
        mu_index: int
            Motor Unit the peak belongs to, to keep dict format consistent.
        l: int
            One half of the action potential discharge time in samples.
            Default value of 31 corresponds to approximately 15 ms at a
            sampling rate of 2048 Hz.

    Returns
    -------
        dict
            Dictionary containing shapes for a given peak per channel.
    """
    raw = flatten_signal(raw)
    channels = raw.shape[0]
    shape_dict = {}

    low = peak - l
    high = peak + l + 1

    shape = raw[:, low:high]  # Shape is channels x Firings; frequently 63 x 62
    # Each of 62 values is a signal

    # Make dictionary from this data

    # Create channel index of each peak
    channel_index = np.repeat(np.arange(channels), l * 2 + 1)  
    # 64 zeros,
    # 64 ones,
    # 64 twos,
    # [...],
    # 64 sixty-threes.

    # Get sample number of each position along each peak
    sample = np.arange(l * 2 + 1)
    sample = np.tile(sample, channels)
    # sample <- [0,1,2,...,61,0,1,2,...,61]

    # Get signals of each peak
    signal = shape.flatten()
    if peak < l:
        neg_idx = abs(peak - l)
        signal[:neg_idx] = np.repeat(0, neg_idx)

    shape_dict[f"mu_{mu_index}"] = {
        "sample": sample,
        "signal": signal,
        "channel": channel_index,
    }

    return shape_dict


def channel_preset(preset="standard"):
    """
    Called by muap_plot() function to determine the order of channels when plotting MUAP shapes.

    Parameters
    ----------
    preset: str
        Name of the preset to use.

    Returns
    -------
        dict
            Dictionary containing
                cols: int
                    Number of columns for a given channel arrangement.
                sort_order: list
                    Sort order of all the channels.

    Examples
    --------
        >>> channel_preset(preset='vert63')
        {
        'cols': 5,
        'sort_order': [
            63, 38, 37, 12, 11, 62, 39, 36, 13, 10, 61, 40, 35, 14, 9, 60, 41, 34, 15, 8, 59, 42, 33, 16, 7, 58, 43, 32, 17, 6, 57, 44, 31, 18, 5, 56, 45, 30, 19, 4, 55, 46, 29, 20, 3, 54, 47, 28, 21, 2, 53, 48, 27, 22, 1, 52, 49, 26, 23, 0, 51, 50, 25, 24
            ]
        }
    """

    if preset == "standard":
        sort_order = list(range(0, 64, 1))
        cols = 8

    # Vert 63 preset
    # Note: this is a mirror image of the preset,
    # because the 'empty' channel is on the bottom right
    elif preset == "vert63":
        sort_order = [
            63,
            38,
            37,
            12,
            11,
            62,
            39,
            36,
            13,
            10,
            61,
            40,
            35,
            14,
            9,
            60,
            41,
            34,
            15,
            8,
            59,
            42,
            33,
            16,
            7,
            58,
            43,
            32,
            17,
            6,
            57,
            44,
            31,
            18,
            5,
            56,
            45,
            30,
            19,
            4,
            55,
            46,
            29,
            20,
            3,
            54,
            47,
            28,
            21,
            2,
            53,
            48,
            27,
            22,
            1,
            52,
            49,
            26,
            23,
            0,
            51,
            50,
            25,
            24,
        ]
        cols = 5

    res = dict(cols=cols, sort_order=sort_order)

    return res


def muap_plot(
    mu_data, mu_index, peak_data=None, l=31, peak="", method="RMSE", preset="standard"
):
    """
    Returns a plot for MUAP shapes separated by channel.
    If peak_data is specified, also plots overlay of contribution of the peak to the shape per channel.
    Called by select_peak() function.

    Parameters
    ----------
        mu_data: dict
            Dictionary containing MUAP shapes for each motor unit.
        mu_index: int
            Index of motor unit to examine.
        peak_data: dict
            Dictionary containing shapes for a given peak per channel.
            Specifying it creates the overlay of peak contribution.
        l: int
            One half of action potential discharge time in samples.
            Default value of 31 corresponds to approximately 15 ms at a
            sampling rate of 2048 Hz.
        peak: float:
            Time of the peak, used for the title of the plot.
        method: str
            Metric to use to calculate mean (over all channels) mismatch score
            between averaged shape and given peak.
        preset: str
            Name of preset to use, for arranging the channels on the plot.

    Returns
    -------
        altair.vegalite.v4.api.FacetChart
            Facetted altair plot overlaying MU shapes per channel
            and peak shapes per channel.
    """
    alt.data_transformers.disable_max_rows()

    df = pd.DataFrame(mu_data[f"mu_{mu_index}"])
    df["Source"] = "MUAP"
    plot_title = f"MUAP Shapes for MU {mu_index}"
    legend_position = None  # Hide legend when we only showing MUAPs
    sort_order = channel_preset(preset)["sort_order"]
    cols = channel_preset(preset)["cols"]

    # If we are passed peak data, that means a peak was selected for analysis
    # So we will add a layer of contribution of each peak to the shape by channel
    if peak_data:
        peak_df = pd.DataFrame(peak_data[f"mu_{mu_index}"])
        peak_df["Source"] = "Peak Contribution"
        df = pd.concat([df, peak_df])
        err = mismatch_score(mu_data, peak_data, mu_index, method=method, channel=-1)
        err = round(err)
        plot_title = f"Peak at {peak} s contribution per Channel to MU {mu_index}. RMSE = {err}"  # And change the plot title to include the peak index and RMSE

        legend_position = alt.Legend(
            orient="none",
            title=None,
            legendX=400,
            legendY=-40,
            direction="horizontal",
            titleAnchor="middle",
        )  # Show Legend when showing overlay

    selection = alt.selection_multi(fields=["Source"], bind="legend")

    # Main MUAP plot
    plot = (
        alt.Chart(df, title=plot_title)
        .encode(
            x=alt.X("sample", axis=None),
            y=alt.Y("signal", axis=None),
            color=alt.Color(
                "Source",
                scale={"range": ["#fd3a4a", "#99a7f1"]},
                legend=legend_position,
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            facet=alt.Facet(
                "channel",
                columns=cols,
                spacing={"row": 0},
                header=alt.Header(
                    titleFontSize=0,
                    labelFontSize=14,
                ),
                sort=sort_order,
            ),
        )
        .mark_line()
        .properties(width=112, height=100)
        .configure_title(fontSize=14, anchor="middle")
        .configure_axis(labelFontSize=14)
        .configure_view(strokeWidth=0)
        .add_selection(selection)
    )

    return plot


def pulse_plot(pt, c_sq_mean, mu_index, sel_type="single"):
    """
    Plot firings and firing rate for a given motor unit.

    Parameters
    ----------
        pulse_train: np.array
            Motor unit pulse train.
        c_sq_mean: np.array
            Centered, squared and averaged firings over the duration of the trial.
        mu_index: int
            Motor unit of interest to plot firings for.
        sel_type: str
            Whether to select single points or intervals.

    Returns
    -------
        altair.vegalite.v4.api.VConcatChart
            Plots containing instantaneous firing rate plot,
            signal strength plots, and overlay between the two.
    """

    color_pulse = "#35d3da"
    color_rate = "#9cb806"

    mu_count = pt.squeeze().shape[0]

    motor_df = pd.DataFrame(columns=["Pulse", "Strength", "Motor Unit", "Hz"])

    for i in range(0, mu_count):
        # PT for MU of interest:
        pt_selected = pt.squeeze()[i].squeeze()
        strength_selected = c_sq_mean[pt_selected]
        hertz = np.insert(1 / np.diff(pt_selected) * 2048, 0, 0)

        # Make those into DF:
        pulses_i = {
            "Pulse": pt_selected,
            "Strength": strength_selected,
            "Motor Unit": i,
            "seconds": pt_selected / 2048,
            "Hz": hertz,
        }
        motor_df_i = pd.DataFrame(pulses_i)
        motor_df = pd.concat([motor_df, motor_df_i])

        motor_df = motor_df.loc[motor_df["Motor Unit"] == mu_index]

    # Single peak selection for signal and frequency plots
    sel_peak = alt.selection_single(name="sel_peak")

    # Interval along x-axis selection for the top plot for close-up purposes
    sel_interval = alt.selection_interval(encodings=["x"], name="sel_interval")

    chart_top_base = (
        alt.Chart(motor_df)
        .encode(
            alt.X(
                "seconds:Q",
                axis=alt.Axis(title="Time (s)", grid=False),
            )
        )
        .properties(width=1000, height=100)
    )

    # Create rate layer for the top nav chart
    chart_top_rate = (
        chart_top_base.mark_point(size=30, color=color_rate)
        .encode(
            alt.Y(
                "Hz:Q",
                axis=alt.Axis(
                    title="Instantaneous Firing Rate (Hz)",
                    grid=False,
                    format=".0f",
                    titleColor=color_rate,
                ),
            )
        )
        .add_selection(sel_interval)
    )

    # Create pulse layer for the top nav chart
    chart_top_pulse = chart_top_base.mark_bar(
        size=3.5, color=color_pulse, opacity=0.3
    ).encode(
        alt.Y(
            "Strength:Q",
            axis=alt.Axis(
                title="Signal (A.U.)", grid=False, format="s", titleColor=color_pulse
            ),
        )
    )

    # Combine pulse and top on the same chart with two y-axis
    chart_top = alt.layer(chart_top_pulse, chart_top_rate).resolve_scale(
        y="independent"
    )

    # Main rate chart with peak selection
    chart_rate = (
        alt.Chart(motor_df)
        .encode(
            alt.X(
                "seconds:Q",
                axis=alt.Axis(title="Time (s)", grid=False),
                scale=alt.Scale(domain=sel_interval),
            ),
            alt.Y(
                "Hz:Q",
                axis=alt.Axis(
                    title="Instantaneous Firing Rate (Hz)",
                    grid=False,
                    format=".0f",
                    titleColor=color_rate,
                ),
            ),
            color=alt.condition(
                sel_peak, alt.value(color_rate), alt.value("lightgray"), legend=None
            ),
            tooltip=[
                alt.Tooltip("Hz", format=".2f"),
                alt.Tooltip("seconds", format=".2f"),
            ],
        )
        .properties(width=1000, height=250)
        .mark_point(size=30)
        .add_selection(sel_peak)
        .transform_filter(sel_interval)
    )

    # Main pulse chart with peak selection
    chart_pulse = (
        alt.Chart(motor_df)
        .encode(
            alt.X(
                "seconds:Q",
                axis=alt.Axis(title="Time (s)", grid=False),
                scale=alt.Scale(domain=sel_interval),
            ),
            alt.Y(
                "Strength:Q",
                axis=alt.Axis(
                    title="Signal (A.U.)",
                    grid=False,
                    format="s",
                    titleColor=color_pulse,
                ),
            ),
            color=alt.condition(
                sel_peak, alt.value(color_pulse), alt.value("lightgray"), legend=None
            ),
        )
        .mark_bar(size=3.5)
        .add_selection(sel_peak)
        .properties(width=1000, height=250)
        .transform_filter(sel_interval)
    )

    return chart_top & chart_rate & chart_pulse


def select_peak(
    selection, mu_index, raw, shape_dict, pt, preset="standard", method="RMSE"
):
    """
    Retrieves a given peak (if any) and re-graphs MUAP plot via muap_plot() function.
    Called within dashboard() function, binded to the peak selection on pulse graphs.

    Parameters
    ----------
        selection: array
            Selection object to dig into and retrieve peak index to plot.
        mu_index: int
            Currently plotted motor unit.
        raw: numpy.ndarray
            Raw EMG signal array.
        shape_dict: dict
            Dictionary containing MUAP shapes for each motor unit.
        pt: numpy.ndarray
            Multi-dimensional array containing indices of firing times
            for each motor unit.
        preset: str
            Name of preset to use, for arranging the channels on the plot.
        method: str
            Metric to use to calculate mean (over all channels) mismatch score
            between averaged shape and given peak.

    Returns
    -------
        panel.layout.base.Column
            Panel column containing facetted altair plot
            overlaying MU shapes per channel
            and peak shapes per channel.

    """
    global selected_peak

    # If there is no selection, only plot shapes
    # If a peak was selected, plot the overlay
    if not selection:
        plot = muap_plot(shape_dict, mu_index, l=31, preset="standard", method="RMSE")
        selected_peak = -1

    else:
        selected_peak = selection[0] - 1

        # for some reason beyond my grasp these are 1-indexed
        peak = pt.squeeze()[mu_index].squeeze()[selected_peak]

        peak_data = muap_dict_by_peak(raw, peak, mu_index=mu_index, l=31)
        plot = muap_plot(
            shape_dict,
            mu_index,
            peak_data,
            l=31,
            peak=str(round(peak / 2048, 2)),
            preset=preset,
            method=method,
        )

    return pn.Column(
        pn.Row(
            pn.pane.Vega(plot, debounce=10, width=750),
        )
    )


def dashboard(decomp_results, raw, mu_index=0, preset="standard", method="RMSE"):
    """
    Parent function for creating interactive visual component of decomposition.
    Dashboard consists of four plots:
    1. Plot of instantaneous firing rate and signal, primarily for zooming and navigating.
    2. Plot of instantaneous firing rate, which allows for peak selection.
    3. Plot of signal strength, which allows for peak selection.
    4. MUAP plot of individual motor unit shapes by channel, with selected peak overlay.

    Parameters
    ----------
        decomp_results: dict
            Decomposition results.
            Must contain MUPulses key with the motor unit firing indices.

        raw: numpy.ndarray
            Raw EMG data.

        mu_index: int
            Currently plotted Motor Unit.

    Returns
    -------
        panel.layout.base.Column
            Panel object containing interactive altair plots.
    """

    signal = flatten_signal(raw)
    signal = np.apply_along_axis(
        butter_bandpass_filter,
        axis=1,
        arr=signal,
        lowcut=10,
        highcut=900,
        fs=2048,
        order=6,
    )
    centered = center_matrix(signal)
    c_sq = centered ** 2
    c_sq_mean = c_sq.mean(axis=0)

    pt = decomp_results["MUPulses"]

    shape_dict = muap_dict(raw, pt, l=31)
    pulse = pulse_plot(pt, c_sq_mean, mu_index, sel_type="interval")
    pulse_pn = pn.pane.Vega(pulse, debounce=10)

    # Below, we bind the selection of the signal or frequency chart to the select_peak function
    # And then bind the rest of the parameters that select_peak expects
    # This will reconstruct muap plot
    mu_charts_pn = pn.bind(
        select_peak,
        pulse_pn.selection.param.sel_peak,
        mu_index,
        raw,
        shape_dict,
        pt,
        preset,
        method,
    )

    # Return column of plots: pulse plots and muap
    res = pn.Column(
        pulse_pn,
        mu_charts_pn,
    )

    return res


def visualize_decomp(decomp_results, raw):
    """
    Wrapper function that allows for cleaner UI for user. Widgets are built within it.

    Parameters
    ----------
        decomp_results: dict
            Decomposition results.
            Must contain MUPulses key with the motor unit firing indices.
        raw: numpy.ndarray
            Raw EMG data.
    Returns
    -------
        panel.layout.base.Column
            Panel object containing four interactive altair plots.
            1. Plot of instantaneous firing rate and signal, primarily for zooming and navigating.
            2. Plot of instantaneous firing rate, which allows for peak selection.
            3. Plot of signal strength, which allows for peak selection.
            4. MUAP plot of individual motor unit shapes by channel, with selected peak overlay.
    """

    # Create widgets

    #  Widget for Motor Unit of interest
    mu_index_widget = pn.widgets.Select(
        name="Motor Unit:",
        options=list(range(len(decomp_results["MUPulses"].squeeze()))),
        value=0,
    )

    #  Widget for preset layout selection
    mu_preset_widget = pn.widgets.Select(
        name="Preset:", options=["standard", "vert63"], value="standard"
    )

    #  Widget for comparison metric selection
    mu_comp_widget = pn.widgets.Select(
        name="Comparison Metric:", options=["RMSE"], value="RMSE"
    )

    # Return widgets and plots
    dash_p = interact(
        dashboard,
        decomp_results=fixed(decomp_results),
        raw=fixed(raw),
        mu_index=mu_index_widget,
        preset=mu_preset_widget,
        method=mu_comp_widget,
    )

    return dash_p
