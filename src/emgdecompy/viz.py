import numpy as np
import pandas as pd
import altair as alt
from emgdecompy.preprocessing import flatten_signal

def muap_dict(raw, pt, l=31):
    """
    Return multi-level dictionary containing sample number, average signal, and channel
    for each motor unit.
    
    Averages the peak shapes over every firing for each MUAP.

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

    for i in range(pt.shape[0]):
        pt[i] = pt[i].squeeze()

        # Create array to contain indices of peak shapes
        ptl = np.zeros((pt[i].shape[0], l * 2), dtype="int")
        
        for j, k in enumerate(pt[i]):
            ptl[j] = np.arange(k - l, k + l)

        ptl = ptl.flatten()
        
        # Create channel index of each peak
        channel_index = np.repeat(np.arange(channels), l * 2)

        # Get sample number of each position along each peak
        sample = np.arange(l * 2)
        sample = np.tile(sample, channels)

        # Get average signals from each channel
        signal = raw[:, ptl].reshape(channels, ptl.shape[0] // (l * 2), l * 2).mean(axis=1).flatten()

        shape_dict[f"mu_{i}"] = {"sample": sample, "signal": signal, "channel": channel_index}
        
    return shape_dict

def muap_plot(shape_dict, mu_index, l=31, page=1, count=12):
    """
    Returns a facetted altair plot of the average MUAP shape from each channel.

    Parameters
    ----------
    shape_dict: dict
        Dictionary returned by muap_dict.
    mu_index: int
        Index of motor unit of interest.
    l: int
        One half of action potential discharge time in samples.
    page: int
        Current page of plots to view. Positive non-zero number.
    count: int
        Number of plots per page. Max is 12.

    Returns
    -------
        altair.vegalite.v4.api.FacetChart
            Facetted altair plot.
    """

    mu_df = pd.DataFrame(shape_dict[f"mu_{mu_index}"])

    row_index = (page - 1) * (l * 2) * count, (page - 1) * (l * 2) * count + (l * 2) * count
    
    if count > 12:
        return "Max plots per page is 12"
    
    # Calculate max number of pages
    last_page =  len(mu_df) // (l * 2) // count + (147 % count > 0)
    
    if page > last_page:
        return f"Last page is page {last_page}."

    plot = (
        alt.Chart(mu_df[row_index[0] : row_index[1]], title="MUAP Shapes")
        .encode(
            x=alt.X("sample", axis=None),
            y=alt.Y("signal", axis=None),
            facet=alt.Facet(
                "channel",
                title=f"Page {page} of {last_page}",
                columns=count / 2,
                header=alt.Header(titleFontSize=14, titleOrient="bottom", labelFontSize=14),
            ),
        )
        .mark_line()
        .properties(width=100, height=100)
        .configure_title(fontSize=18, anchor="middle")
        .configure_axis(labelFontSize=14)
    )

    return plot