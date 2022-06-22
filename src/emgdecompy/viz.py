import numpy as np
import pandas as pd
import altair as alt
import panel as pn
import warnings
from scipy import linalg
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import HTML
import math
from sklearn.metrics import mean_squared_error

pn.extension("vega")


def RMSE(arr1, arr2):
    """
    Evaluates Root Square Error for two series

    Parameters
    ----------
        arr1: iterative object
            First source, likely target.
        arr2: iterative object
            Second source, likely actual values.

    Returns
    -------
        float
            Root Mean Square Error of arr1 vs arr2.

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


def mismatch_score(mu_data, peak_data, mu_index, method=RMSE, channel=-1):
    """
    Evaluates how well a given peak contributes to a given MUAP.
    This is called by muap_plot() function and is used to include error in the title of the muap plot.

    Parameters
    ----------
        mu_data: dict
            Dictionary containing MUAP shapes for each motor unit.
        peak_data: dict
            Dictionary containing shapes for a given peak per channel.
        mu_index: int
            Index of motor unit to examine
        method: function name
            Function to use for evaluating discrepency between mu_data and peak_data.
            Default: RMSE.
        channel: int
            Channel to run evaluation on.
            Defaul = -1 and it means average of all channels.

    Returns
    -------
        float
            Root Mean Square Error of MU data vs Peak data.
    """
    if channel == -1:  # For all channels, we can just
        # straight up compare RMSE across the board
        mu_sig = mu_data[f"mu_{mu_index}"]["signal"]
        peak_sig = peak_data[f"mu_{mu_index}"]["signal"]
        score = RMSE(mu_sig, peak_sig)

    else:  # Otherwise, filter for a given channel
        # filter mu_data for signal data that channel
        indexes = np.where(mu_data[f"mu_{mu_index}"]["channel"] == channel)
        mu_sig = mu_data[f"mu_{mu_index}"]["signal"][indexes]

        indexes = np.where(peak_data[f"mu_{mu_index}"]["channel"] == channel)
        peak_sig = peak_data[f"mu_{mu_index}"]["signal"][indexes]

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
            One half of action potential discharge time in samples.

    Returns
    -------
        dict
            Dictionary containing MUAP shapes for each motor unit.
    """
    raw = flatten_signal(raw)
    channels = raw.shape[0]
    shape_dict = {}
    pt = pt.squeeze()

    for i in range(pt.shape[0]):
        pt[i] = pt[i].squeeze()

        # Create array to contain indices of peak shapes
        ptl = np.zeros((pt[i].shape[0], l * 2 + 1), dtype="int")

        for j, k in enumerate(pt[i]):
            ptl[j] = np.arange(k - l, k + l + 1)

            if np.less(k, l) == True:
                ptl[j] = np.arange(k - l, k + l + 1)
                neg_idx = abs(k - l)
                ptl[j][:neg_idx] = np.repeat(0, neg_idx)

            else:
                ptl[j] = np.arange(k - l, k + l + 1)

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
            One half of action potential discharge time in samples.

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
    channel_index = np.repeat(np.arange(channels), l * 2 + 1)  # 64 zeros,
    # 64 ones,
    # 64 twos,
    # [...],
    # 64 sixty-threes.

    # Get sample number of each position along each peak
    sample = np.arange(l * 2 + 1)
    sample = np.tile(sample, channels)
    # sample <- [0,1,2,...,61,0,1,2,...,61]

    # Get signals of each peak
    signal = (
        shape.flatten()
    )  # TODO: MAKE SURE IT DOESNT FAIL WHEN FIRST FIRING IS BEFORE l=31

    if peak < l:
        neg_idx = abs(peak - l)
        signal[:neg_idx] = np.repeat(0, neg_idx)

    shape_dict[f"mu_{mu_index}"] = {
        "sample": sample,
        "signal": signal,
        "channel": channel_index,
    }

    return shape_dict


def channel_preset(name="standard"):
    """
    Returns a dictionary with two keys:
    'sort_order' with the list to order channels,
    and 'cols' with the number of columns for a given channel arrangement.
    Called by muap_plot() function to determine the order of channels to plot shapes.

    Parameters
    ----------
    name: str
        Name of the preset to use

    Returns
    -------
        dict with two keys
            cols: int
                Number of columns
            sort_order: list
                Sort order of all the channels

    Examples
    --------
        >>> channel_preset(name='vert63')
        {
        'cols': 5,
        'sort_order': [
            63, 38, 37, 12, 11, 62, 39, 36, 13, 10, 61, 40, 35, 14, 9, 60, 41, 34, 15, 8, 59, 42, 33, 16, 7, 58, 43, 32, 17, 6, 57, 44, 31, 18, 5, 56, 45, 30, 19, 4, 55, 46, 29, 20, 3, 54, 47, 28, 21, 2, 53, 48, 27, 22, 1, 52, 49, 26, 23, 0, 51, 50, 25, 24
            ]
        }
    """

    if name == "standard":
        sort_order = list(range(0, 64, 1))
        cols = 8

    elif name == "vert63":
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
    mu_data, mu_index, peak_data=None, l=31, peak="", method=RMSE, preset="standard"
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
            Index of motor unit to examine
        peak_data: dict
            Dictionary containing shapes for a given peak per channel.
            Specifying it creates the overlay of peak contribution
        l: int
            One half of action potential discharge time in samples.
        peak: int:
            Index of the peak, used for the Title of the plot.
        method: function name
            Function to use to calculate mean (over all channels) mismatch score between averaged shape and given peak
        preset: str
            Name of preset to use, for arranging the channels on the plot

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
    legend_position = None  # Hide legend when we only shpwing MUAPs
    sort_order = channel_preset(preset)["sort_order"]
    cols = channel_preset(preset)["cols"]

    if peak_data:
        peak_df = pd.DataFrame(peak_data[f"mu_{mu_index}"])
        peak_df["Source"] = "Peak Contribution"
        df = pd.concat([df, peak_df])
        err = mismatch_score(mu_data, peak_data, mu_index, method=method, channel=-1)
        err = round(err)
        plot_title = (
            f"Peak #{peak} contribution per Channel to MU {mu_index}. RMSE = {err}"
        )
        legend_position = alt.Legend(
            orient="none",
            title=None,
            legendX=400,
            legendY=-40,
            direction="horizontal",
            titleAnchor="middle",
        )  # Show Legend when showing overlay

    selection = alt.selection_multi(fields=["Source"], bind="legend")

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
                    titleOrient="bottom",
                    labelFontSize=14,
                    labelOrient="bottom",
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


def pulse_plot(pt, c_sq_mean, mu_index=None, sel_type="single"):
    """
    Plot firings for a given motor unit.

    Parameters
    ----------
        pulse_train: np.array
            Pulse train.

        c_sq_mean: np.array
            Centered, squared and averaged firings over the duration of the trial.

        mu_index: int
            Motor Unit of interest to plot firings for.
            Default is None and means return all pulses.

    Returns
    -------
        altair plot object
    """

    color_pulse = "#35d3da"
    color_rate = "#9cb806"

    mu_count = pt.shape[0]

    motor_df = pd.DataFrame(columns=["Pulse", "Strength", "Motor Unit", "MS", "Hz"])
    for i in range(1, mu_count + 1):
        # PT for MU of interest:
        pt_selected = pt.squeeze()[i - 1].squeeze()
        strength_selected = c_sq_mean[pt_selected]
        hertz = 1 / np.diff(pt_selected)
        hertz_list = hertz.tolist()
        hertz = [0] + hertz_list
        hz_ms = [x * 2048 for x in hertz]

        # Make those into DF:
        pulses_i = {
            "Pulse": pt_selected,
            "Strength": strength_selected,
            "Motor Unit": i,
            "seconds": pt_selected / 2048,
            "Hz": hz_ms,
        }
        motor_df_i = pd.DataFrame(pulses_i)
        motor_df = pd.concat([motor_df, motor_df_i])

    if mu_index:
        motor_df = motor_df.loc[motor_df["Motor Unit"] == mu_index]
        # brush = alt.selection_interval(encodings=['x'], name='brush') # Don't know if we will use this

    # TODO: Selection only makes sense if we are working with specific MU

    sel_peak = alt.selection_single(name="sel_peak")

    sel_interval = alt.selection_interval(encodings=["x"], name="sel_interval")

    chart_top_base = (
        alt.Chart(motor_df)
        .encode(
            alt.X(
                "seconds:Q",
                axis=alt.Axis(title="Time (sec)", grid=False),
            )
        )
        .properties(width=1000, height=100)
    )

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

    chart_top = alt.layer(chart_top_pulse, chart_top_rate).resolve_scale(
        y="independent"
    )

    chart_rate = (
        alt.Chart(motor_df)
        .encode(
            alt.X(
                "seconds:Q",
                axis=alt.Axis(title="Time (sec)", grid=False),
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
            # scale=alt.Scale(domain=(0, motor_df['seconds']motor_df.loc[motor_df['seconds'].idxmax()]['Hz']))),
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

    chart_pulse = (
        alt.Chart(motor_df)
        .encode(
            alt.X(
                "seconds:Q",
                axis=alt.Axis(title="Time (sec)", grid=False),
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


def create_widget_dd(options, value=0, desc="Motor Unit:", disabled=False):
    """
    Create a dropdown widget.

    Parameters
    ----------
        options: list
            Options for the dropdown.
        value: int or str
            Original value to be selected.
        desc: str
            Description to be displayed above the widget.
        disabled: bool
            Whether the widget is disabled by default

    Returns
    -------
        widget object: dropdown widget to be used in altair interactions.
    """

    widget = widgets.Dropdown(
        options=options,
        value=value,
        description=desc,
        disabled=disabled,
    )

    return widget


def select_peak(selection, mu_index, raw, shape_dict, pt):
    """
    Interactivity function for the Firing plot.
    Retrieves a given peak (if any) and re-graphs MUAP plot via muap_plot() function.
    Called within dashboard() function, binded to the peak selection on pulse graphs.

    Parameters
    ----------
        selection: selection object
            Selection object to dig into and retrieve peak index to plot.

        mu_index: int
            Currently plotted Motor Unit.

        raw: numpy.ndarray
            Raw EMG signal array.

        shape_dict: dict
            Dictionary containing MUAP shapes for each motor unit.

        pt: numpy.ndarray
            Multi-dimensional array containing indices of firing times
            for each motor unit.

    Returns
    -------
        altair plot object

    """
    if not selection:
        plot = muap_plot(shape_dict, mu_index, l=31)
        # return '## No selection'

    else:
        sel = selection[0] - 1
        # for some reason beyond my grast these are 1-indexed
        peak = pt[mu_index][sel]
        # peak = pt[mu_index][0]

        peak_data = muap_dict_by_peak(raw, peak, mu_index=mu_index, l=31)
        plot = muap_plot(shape_dict, mu_index, peak_data, l=31, peak=str(peak))

    return pn.Column(
        pn.Row(
            pn.pane.Vega(plot, debounce=10, width=750),
        )
    )


def remove_false_peak(decomp_results, mu_index, peak):
    """
    Removes a false positive peak from MU Pulses from the dictionary
    from decomposition results.

    Parameters
    ----------
        decomp_results: dict
            Dict object outputted by the decomposition() function.
        mu_index: int
            Motor unit index for which to remove the false peak.
        peak: int
            Peak timing to remove.

    Returns
    -------
        dict
            Amended results dictionary.
    """

    decomp_results["MUPulses"] = list(decomp_results["MUPulses"])
    decomp_results["MUPulses"][0][mu_index] = np.delete(
        decomp_results["MUPulses"][0][mu_index],
        np.argwhere(decomp_results["MUPulses"][0][mu_index][0] == peak),
    )
    decomp_results["MUPulses"] = np.array(decomp_results["MUPulses"], dtype=object)

    return decomp_results


def dashboard(pt, c_sq_mean, mu_index, raw, shape_dict):
    """
    Parent function for creating interactive visual component of decomposition.
    Dashboard consists of four plots:
    1. Plot of firings and signal, primarily for zooming and navigating.
    2. Plot of signal strength, which allows for peak selection.
    3. Plot of firings, which allows for peak selection.
    4. MUAP plot of individual motor unit shapes by channel, with selected peak overlay

    Parameters
    ----------
        pt: numpy.ndarray
            Multi-dimensional array containing indices of firing times
            for each motor unit.

        c_sq_mean: np.array
            Centered, squared and averaged firings over the duration of the trial.

        mu_index: int
            Currently plotted Motor Unit.

        raw: numpy.ndarray
            Raw EMG signal array.

        shape_dict: dict
            Dictionary containing MUAP shapes for each motor unit.

    Returns
    -------
        panel object containing interactive altair plots
    """

    pulse = pulse_plot(pt, c_sq_mean, mu_index, sel_type="interval")
    vega_pane = pn.pane.Vega(pulse, debounce=10)
    return pn.Column(
        vega_pane,
        pn.bind(
            select_peak,
            vega_pane.selection.param.sel_peak,
            mu_index,
            raw,
            shape_dict,
            pt,
        ),
    )
