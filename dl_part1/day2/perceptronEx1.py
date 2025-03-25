import numpy as np
import torch

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.3

    tp = np.sum(w * x) + b

    if tp<=0:
        return 0
    else:
        return 1

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7

    tp = np.sum(w*x)+b

    if tp<=0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7

    tp = np.sum(w*x)+b

    if tp<=0:
        return 0
    else:
        return 1

xdata = [[0,0],[0,1],[1,0],[1,1]]

for data in xdata:
    print(f'{data} -> {AND(*data)}')