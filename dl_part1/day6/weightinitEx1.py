import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)

input_data = np.random.randn(1000, 100)

node_num = 100
hidden_layer_size = 5
activations = {}
x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    #w = np.random.randn(node_num, node_num)
    #w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / (node_num + node_num)) #xavier
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 * 2 / (node_num + node_num)) #xavier
    a = np.dot(x, w)
    #output = sigmoid(a)
    output = Relu(a)
    activations[i] = output

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) +'-layer')
    if i != 0:
        plt.yticks([],[])
    plt.hist(a.flatten(), 30, range=(0,1))
    plt.xlim(0.1, 1)
    plt.ylim(0, 7000)
plt.show()
