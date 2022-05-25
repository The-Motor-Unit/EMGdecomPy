from emgdecompy import decomposition as emg
import numpy as np

def test_normalize():
    """
    Run unit test on normalize function from EMGdecomPy.
    """
    for i in range(0, 10):

        # initialize array
        x = np.random.randint(1, 1000)
        y = np.random.randint(1, 1000)

        fake_data = np.random.rand(x, y)

        # calculate Frobenius norm manually
        squared = fake_data ** 2
        summed = squared.sum()
        frob_norm = np.sqrt(summed)

        # this is how the normalize() calculates Frobenius norm
        fx_norm = np.linalg.norm(fake_data)

        assert np.allclose(frob_norm, fx_norm), "Frobenius norm incorrectly calculated."

        # normalize
        normalized = fake_data / frob_norm

        fx = emg.normalize(fake_data)

        assert np.allclose(normalized, fx), "Array normalized incorrectly."
        
def test_orthogonalize():
    """
    Run unit tests on orthogonalize() function from EMGdecomPy.
    """
    for i in range(0, 10):
        x = np.random.randint(1, 100)
        y = np.random.randint(1, 100)
        z = np.random.randint(1, 100) 

        w = np.random.randn(x, y) * 1000 
        b = np.random.randn(x, z) * 1000

        assert b.T.shape[1] == w.shape[0], "Dimensions of input arrays are not compatible."

        # orthogonalize by hand 
        ortho = w - b @ (b.T @ w)

        fx = emg.orthogonalize(w, b)

        assert np.array_equal(ortho, fx), "Manually calculated array not equivalent to emg.orthogonalize()"
        assert fx.shape == w.shape, "The output shape of w is incorrect."