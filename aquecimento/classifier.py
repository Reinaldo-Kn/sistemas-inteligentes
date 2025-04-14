# -*- coding: utf-8 -*-
"""
Exemplo de preditor linear como classificador

@author: Prof. Daniel Cavalcanti Jeronymo
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Simple binary function
def binary(x):
    return 1 if x >= 0.5 else 0

# Linear predictor
def f(x, w, b):
    return np.sum(np.array(x)*np.array(w)) + b

# Binary classifier
def fw(x, w, b):
    return binary(f(x, w, b))

# calculate MSE (mean squared error) for a training data set
def loss(x, y, w, b):
    mse = 0
    n = len(y)
    
    for input, output in zip(x, y):
        error = fw(input, w, b) - output
        mse += math.pow(error, 2)/n
        
    return mse
    
# Trains a linear classifier
# stochastic search for values of weights and bias until MSE is zero
def train(x, y):
    mse = 1
    n = len(x)
    
    while mse != 0:
        w = np.random.normal(0, 2, 1) # random guess
        b = np.random.normal(0, 2, 1) # random guess
        
        mse = loss(x, y, w, b)
        
    return (w, b)

# Arbitrary data set
# input dataset
x = np.array([0, 1, 2, 3, 5, 6, 7, 8])/8
# output dataset
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Learn a linear classifier 
w, b = train(x, y)

# Plot decision boundary and predictions
plt.plot(x, y, 'ro')
yb =  [f(xi, w, b) for xi in x]
yp =  [fw(xi, w, b) for xi in x]
plt.plot(x, yb, 'k--')
plt.plot(x, yp)
plt.show()

print(w)
print(b)