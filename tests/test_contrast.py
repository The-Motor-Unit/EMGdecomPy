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

def test_log_cosh():
    """
    Run unit tests on log_cosh() function from EMGdecomPy.
    """
    
    for i in range (0, 5):

        x = np.random.randint(0, 709) 
        assert x <= 709, "X is too large, will result in calculation overflow."

        # test base contrast function, log(cosh(x))
        x_cosh = 1/2 * (np.exp(x) + np.exp(-x))  # manually calculate cosh(x) 
        x_log = np.log(x_cosh)

        assert x_log == emg.log_cosh(x), "Base contrast function incorrectly calculated."

        # test first derivative of contrast function, tanh(x)
        x_tanh = np.sinh(x)/np.cosh(x) # manually calculate tanh(x)

        assert np.isclose(x_tanh, emg.log_cosh(x, der=True)), "1st derivative fx incorrectly calculated."

