import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


x = np.arange(-10,10,0.1)
y = sigmoid(x)

plt.plot([0,0],[1,0],':')
plt.plot([-10,10],[0,0],':')
plt.plot([-10,10],[1,1],':')

# y1 = sigmoid(0.5*x)
# y2 = sigmoid(x)
# y3 = sigmoid(2*x)

y1 = sigmoid(x-1.5)
y2 = sigmoid(x)
y3 = sigmoid(x+1.5)


plt.plot(x,y1,label='x-1.5')
plt.plot(x,y2,label='x')
plt.plot(x,y3,label='x+1.5')
plt.legend(loc='best')
plt.grid(True)
plt.show()