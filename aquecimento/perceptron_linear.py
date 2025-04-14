# -*- coding: utf-8 -*-
"""
Exemplo de perceptron linear para funções lógicas

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Heaviside step function
@np.vectorize
def heaviside(x):
    return 1.0 if x > 0.0 else 0.0

# linear perceptron
def perceptron(x, w, b):
    return heaviside(np.sum(np.array(x)*np.array(w)) + b)

# helper function for decision boundary plotting
def perceptron3d(x, y, w, b):
    return heaviside(x*w[0] + y*w[1] + b)#x*w[0] + y*w[1] + b

# calculate MSE (mean squared error) for a training data set
def loss(x, y, w, b):
    mse = 0
    n = len(y)
    
    for input, output in zip(x, y):
        error = perceptron(input, w, b) - output
        mse += math.pow(error, 2)/n
        
    return mse

# stochastic search for values of weights and bias until MSE is zero
def train(x, y):
    mse = 1
    
    w = 0
    b = 0
    
    while mse != 0:
        w = np.random.normal(0, 2, len(x[0]))
        b = np.random.normal(0, 2, 1)
        
        mse = loss(x, y, w, b)
        
    return (w, b)

# input / output datasets
# AND logical gate
# x = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([0, 0, 0, 1])        
# NAND logical gate
# x = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([1, 1, 1, 0])
# OR logical gate
# x = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([0, 1, 1, 1])

# XOR logical gate
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])

# Make the perceptron learn 
w, b = train(x, y)

# 3D plot of training data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y, c='b', marker='o')

# Plot decision boundary
xx1 = np.linspace(0, 1, 100)
xx2 = np.linspace(0, 1, 100)
x, y = np.meshgrid(xx1, xx2)
z = np.vectorize(perceptron3d, excluded=['w', 'b'])(x=x, y=y, w=w, b=b)
zlow = np.copy(z)
zhigh = np.copy(z)
zlow[z<=0] = np.nan
zhigh[z>0] = np.nan
ax.plot_surface(x, y, zlow, color='r', alpha=0.2)
ax.plot_surface(x, y, zhigh, alpha=0.2)

# Set labels, view and show
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.view_init(90, 0)
plt.show()

print(w)
print(b)
