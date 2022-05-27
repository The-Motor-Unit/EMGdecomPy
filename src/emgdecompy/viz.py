import numpy as np
import pandas as pd
import altair as alt
from emgdecompy.preprocessing import flatten_signal

def muap_dict(raw, pt, l):
    """
    Return multi-level dictionary containing sample number, signal, and peak index
    for each motor unit.
    
    Averages the peak shapes along all channels for each MUAP.

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

    shape_dict = {}

    for i in range(pt.shape[0]):
        pt[i] = pt[i].squeeze()

        # Create array to contain indices of peak shapes
        ptl = ptl = np.zeros((pt[i].shape[0], l * 2), dtype="int")

        # Get sample number of each position along each peak
        sample = np.arange(l * 2)

        # Create index of each peak
        peak_index = np.zeros((pt[i].shape[0], l * 2), dtype="int")

        for j, k in enumerate(pt[i]):
            ptl[j] = np.arange(k - l, k + l)

            peak_index[j] = np.full(l * 2, j)

        ptl = ptl.flatten()

        peak_index = peak_index.flatten()

        # Get sample number of each position along each peak
        sample = np.arange(l * 2)
        sample = np.tile(sample, pt[i].shape[0])

        # Get signals of each peak
        signal = raw[:, ptl].mean(axis=0).reshape(pt[i].shape[0], l * 2).flatten()

        shape_dict[f"mu_{i}"] = {"sample": sample, "signal": signal, "peak": peak_index}

    return shape_dict

def muap_plot(shape_dict, mu_index, page=0, count=12):
    """
    Returns a facetted altair plot of the average MUAP shapes for each MUAP.

    Parameters
    ----------
    shape_dict: dict
        Dictionary returned by muap_dict.
    mu_index: int
        Index of motor unit of interest.
    page: int
        Current page of plots to view.
    count: int
        Number of plots per page. Max is 12.

    Returns
    -------
        altair.vegalite.v4.api.FacetChart
            Facetted altair plot.
    """

    mu_df = pd.DataFrame(shape_dict[f"mu_{mu_index}"])

    row_index = page * (l * 2) * count, page * (l * 2) * count + (l * 2) * count
    
    if count > 12:
        return "Max plots per page is 12"
    
    if row_index[0] > len(mu_df):
        return "No more pages."

    plot = (
        alt.Chart(mu_df[row_index[0] : row_index[1]], title="MUAP Shapes")
        .encode(
            x=alt.X("sample", axis=None),
            y=alt.Y("signal", axis=None),
            facet=alt.Facet(
                "peak",
                title=None,
                columns=firing_number / 2,
                header=alt.Header(labelFontSize=14),
            ),
        )
        .mark_line()
        .properties(width=100, height=100)
        .configure_title(fontSize=18, anchor="middle")
        .configure_axis(labelFontSize=14)
    )

    return plot