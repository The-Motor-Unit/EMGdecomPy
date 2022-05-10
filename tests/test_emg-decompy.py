from emg-decompy import emg-decompy

R = 5
x = np.array([1,2,3,4,5,6,7])
assert extend_input_by_R(x, R)[0][0] == x[0]
assert extend_input_by_R(x, R)[0][-1] == 0
assert len(extend_input_by_R(x, R)) == len(x)
assert extend_input_by_R(x, R)[-1][0] == x[-1]
assert extend_input_by_R(x, R)[0][0] == extend_input_by_R(x, R)[1][1]

# +
R = 5
x_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

assert len(extend_input_all_channels(x_mat, R)) == len(x_mat)
assert len(extend_input_all_channels(x_mat, R)[0][0]) == R + 1
assert extend_input_all_channels(x_mat, R)[0][0][0] == extend_input_all_channels(x_mat, R)[0][1][1]
assert extend_input_all_channels(x_mat, R)[0][0][-1] == 0
assert sum(extend_input_all_channels(x_mat, R)[-1][-1]) == sum(x_mat[-1])
