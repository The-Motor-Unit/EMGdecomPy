R_one = 5
R_two = 10
x = np.array([1, 2, 3, 4, 5, 6, 7])

# Tests for extend_input_by_R

assert extend_input_by_R(x, R_one)[0][0] == x[0]
assert extend_input_by_R(x, R_one)[0][-1] == 0
assert extend_input_by_R(x, R_one).shape == (len(x), R_one + 1)
assert extend_input_by_R(x, R_one)[-1][0] == x[-1]
assert extend_input_by_R(x, R_one)[0][0] == extend_input_by_R(x, R_one)[1][1]

assert extend_input_by_R(x, R_two)[0][0] == x[0]
assert extend_input_by_R(x, R_two)[0][-1] == 0
assert extend_input_by_R(x, R_two).shape == (len(x), R_two + 1)
assert extend_input_by_R(x, R_two)[-1][0] == x[-1]
assert extend_input_by_R(x, R_two)[0][0] == extend_input_by_R(x, R_two)[1][1]

# Tests for extend_input_all_channels

x_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

assert extend_input_all_channels(x_mat, R_one).shape == (len(x_mat), len(x_mat[0]), R_one + 1)
assert extend_input_all_channels(x_mat, R_one)[0][0][0] == extend_input_all_channels(x_mat, R_one)[0][1][1]
assert extend_input_all_channels(x_mat, R_one)[0][0][-1] == 0
assert sum(extend_input_all_channels(x_mat, R_one)[-1][-1]) == sum(x_mat[-1])

assert extend_input_all_channels(x_mat, R_two).shape == (len(x_mat), len(x_mat[0]), R_two + 1)
assert extend_input_all_channels(x_mat, R_two)[0][0][0] == extend_input_all_channels(x_mat, R_two)[0][1][1]
assert extend_input_all_channels(x_mat, R_two)[0][0][-1] == 0
assert sum(extend_input_all_channels(x_mat, R_two)[-1][-1]) == sum(x_mat[-1])