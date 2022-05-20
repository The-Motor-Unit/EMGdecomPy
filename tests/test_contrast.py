from emgdecompy import contrast as emg
import numpy as np

def test_skew():
    """
    Run unit tests on skew() function from EMGdecomPy.
    """
    for i in range(0, 5):

        x = np.random.randint(0, 1000)
        assert x % 1 == 0, "Input must be an integer."

        # test base version of contrast function 
        no_der = np.power(x, 3) / 3
        func_no_der = emg.skew(x)

        assert no_der == func_no_der, "Contrast function incorrectly applied."

       # test first derivative of contrast function 
        first_der = x * x  # first derivative of x^3/3 = x^2
        func_der = emg.skew(x, der=True)

        assert first_der == func_der, "First derivative not calculated correctly."