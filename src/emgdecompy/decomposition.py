import numpy as np
from preprocessing import flatten_signal, extend_all_channels, whiten
from separation import separation
from refinement import refinement

def decomposition(
    x, M=64, Tolx=10e-4, fun=skew, max_iter_sep=10, th_sil=0.9, filepath="", max_iter_ref=10
):
    """
    Main function duplicating decomposition algorithm from Negro et al. (2016).
    Performs decomposition of raw EMG signals.

    Parameters
    ----------
        x: numpy.ndarray
            The input matrix.
        Tolx: float
            Tolx for element-wise comparison in separation.
        fun: function
            Contrast function to use.
            skew, og_cosh or exp_sq
        max_iter_sep: int > 0
            Maximum iterations for fixed point algorithm.
            When to stop if it doesn't converge.
        th_sil: float
            Silhouette score threshold for accepting a separation vector.
        max_iter_ref: int > 0
            Maximum iterations for refinement.
        filepath: str
            Filepath/name to be used when saving pulse trains.

    Returns
    -------
        numpy.ndarray
            Decomposed matrix B.

    Examples
    --------
    >>> x = gl_10 = loadmat('../data/raw/gl_10.mat') #Classic gold standard data
    >>> x = gl_to['SIG']
    >>> decomposition(x)

    """
    # Flatten
    x = flatten_signal(x)

    # Extend
    x_ext = extend_all_channels(x, 10)

    # Subtract mean + Whiten
    z = whiten(x_ext)

    B = np.zeros((z.shape[0], z.shape[0]))

    for i in range(M):

        # Separate
        w_i = separation(z, B, Tolx, fun, max_iter_sep)

        # Refine
        B[:i] = refinement(w_i, z, i, max_iter_ref, th_sil, filepath)

    return B