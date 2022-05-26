from emgdecompy import contrast as emg
import numpy as np

def test_skew():
    """
    Run unit tests on skew() function from EMGdecomPy.
    """
    for i in range(0, 10):
        x = np.random.randint(1,1000)
        y = np.random.randint(1,1000)

        test_arr = np.random.choice(np.arange(-1000, 1000), size=(x, y)) # np.random.rand(x, y)

        # test base version of contrast function 
        manual_calc = np.power(test_arr, 3) / 3 # calculate by hand 
        emg_no_deriv = emg.skew(test_arr)    

        assert np.array_equal(manual_calc, emg_no_deriv), "Contrast function incorrectly applied."

        # reverse calculate initial values in input 
        reverse_calc = np.cbrt(emg_no_deriv[-1][-1] * 3)
        
        assert np.isclose(reverse_calc, test_arr[-1][-1]), "Contrast function incorrectly applied."

        # test first derivative of contrast function 
        manual_calc_deriv = np.power(test_arr, 2) # first derivative of x^3/3 = x^2
        emg_deriv = emg.skew(test_arr, der=True)

        assert np.allclose(manual_calc_deriv, emg_deriv), "First derivative not calculated correctly."

        # reverse calculate initial values in input 
        reverse_calc_der = np.sqrt(emg_deriv[-1][-1])
        abs_test_value = abs(test_arr[-1][-1]) # absolute because +/- is lost in back calculation
        
        assert np.isclose(reverse_calc_der, abs_test_value), "Contrast function incorrectly applied."

def test_log_cosh():
    """
    Run unit tests on log_cosh() function from EMGdecomPy.
    """
    for i in range (0, 5):

        # test values between upper and lower limit of 710
        x = np.random.randint(1,100)
        y = np.random.randint(1,100)
        test_arr = np.random.choice(np.arange(-709, 709), size=(x, y)) 

        # test base contrast function, log(cosh(x))
        x_cosh = 1/2 * (np.exp(test_arr) + np.exp(-test_arr))  # manually calculate cosh(x) 
        x_log = np.log(x_cosh)

        assert np.isclose(x_log, emg.log_cosh(test_arr)), "Base contrast function incorrectly calculated."

        # test first derivative of contrast function, tanh(x)
        x_tanh = np.sinh(x)/np.cosh(x) # manually calculate tanh(x)

        assert np.isclose(x_tanh, emg.log_cosh(x, der=True)), "Firstt derivative contrast function incorrectly calculated."

    # test edge cases (values +/- 710)
    z = np.array([[710, 811, 900],[-710, -811, -900]])
    z_log_cosh = np.array([[709.3, 810.3, 899.3],[709.3, 810.3, 899.3]])
    z_first_deriv = np.array([[1, 1, 1],[-1, -1, -1]])

    assert np.allclose(emg.log_cosh(z), z_log_cosh), "Edge cases not properly handled."
    assert np.allclose(emg.log_cosh(z, der=True), z_first_deriv), "Edge cases not properly handled."


def test_exp_sq():
    """
    Run unit tests on exp_sq() function from EMGdecomPy.
    """
    for i in range (0, 10):

        x = np.random.randint(1,1000)
        y = np.random.randint(1,1000)
        test_arr = np.random.choice(np.arange(-1000, 1000), size=(x, y)) 

        # base function = exp((-x^2/2))
        fx = - np.power(test_arr, 2) / 2 # calculate inner part of function

        # test base contrast function, no derivative
        exp_fx = np.exp(fx)
        no_deriv = emg.exp_sq(test_arr)
        
        assert np.count_nonzero(exp_fx) == np.count_nonzero(no_deriv),"Base contrast function incorrectly calculated."
        assert np.argmax(exp_fx) == np.argmax(no_deriv),"Base contrast function incorrectly calculated."
        assert np.array_equal(exp_fx, no_deriv), "Base contrast function incorrectly calculated."

        # test first derivative of base contrast function using exponent power rule
        der_fx = np.exp(fx / 2) * np.exp(fx / 2) * -test_arr 
        first_deriv = emg.exp_sq(test_arr, der=True)
        
        assert np.count_nonzero(der_fx) == np.count_nonzero(first_deriv), "First derivative function incorrectly calculated."
        assert np.argmax(der_fx) == np.argmax(first_deriv),"First derivative function incorrectly calculated."
        assert np.allclose(der_fx, first_deriv), "First derivative function incorrectly calculated."
