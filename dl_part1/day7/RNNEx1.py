import numpy as np

time_step = 10
input_size = 4
hidden_size = 8

inputs = np.random.random((time_step, input_size))
print(inputs.shape)
hidden_state_t = np.zeros((hidden_size,))
print(hidden_state_t.shape)

wx = np.random.random((input_size, hidden_size))
wh = np.random.random((hidden_size, hidden_size))
b = np.random.random((hidden_size,))
print()
print(wx.shape)
print(wh.shape)
print(b.shape)

total_hidden_state = []

for input_t in inputs:
    output_t = np.tanh(np.dot(input_t, wx) + np.dot(hidden_state_t, wh) + b)
    total_hidden_state.append(list(output_t))
    hidden_state_t = output_t

print()
total_hidden_state = np.stack(total_hidden_state, axis=0)
print(total_hidden_state.shape)
print(total_hidden_state)



