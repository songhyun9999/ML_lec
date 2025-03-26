import numpy as np

wsum = np.array([0.9, 2.9, 4.0])

def softmax(ws):
    exp_a = np.exp(ws)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def softmax2(ws):
    c = np.max(ws)
    exp_a = np.exp(ws - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

output = softmax(wsum)
print(output)
print(output.sum())

output2 = softmax2(wsum)
print(output2)
print(output2.sum())
