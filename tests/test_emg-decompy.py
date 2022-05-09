from emg-decompy import emg-decompy

R = 5
x = np.array([1,2,3,4,5,6,7])
assert extend_input_by_R(x, R)[0][0] == x[0]
assert extend_input_by_R(x, R)[0][-1] == 0
assert len(extend_input_by_R(x, R)) == len(x)
assert extend_input_by_R(x, R)[-1][0] == x[-1]
assert extend_input_by_R(x, R)[0][0] == extend_input_by_R(x, R)[1][1]


