#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LINMA1731 Stochastic Processes

Code for the project

@author: Philémon Beghin and Jehum Cho
"""
from matplotlib import cm

"""
LORENZ SYSTEM
"""

# from https://en.wikipedia.org/wiki/Lorenz_system

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi
import random


def L1(X, Y):
    distance = abs(X - Y)
    return np.sum(distance)


def Linf(X, Y):
    distance = abs(X - Y)
    return np.max(distance)


def makeLorentz(params):
    def Lorenz(state, t):
        x, y, z = state  # Unpack the state vector
        return params[0] * (y - x), x * (params[1] - z) - y, x * y - params[2] * z  # Derivatives

    return Lorenz


n = 100
distancesL1 = np.zeros(n)
distancesLinf = np.zeros(n)
params = np.zeros((100, 3))
params[:, 0] = np.linspace(10, 5, n)
params[:, 1] = np.linspace(28, 28, n)
params[:, 2] = np.linspace(8 / 3, 8 / 3, n)

state0 = [1.0, 1.0, 1.0]  # initial condition
t = np.arange(0.0, 100.0, 0.02)  # time vector
states0 = odeint(makeLorentz(params[0]), state0, t)  # vector containing the (x,y,z) positions for each time step
squares = np.int_(np.floor(states0 / 5))
matrix0 = np.zeros((8, 12, 10))
for elem in squares:
    square = tuple(elem)
    x, y, z = square
    matrix0[x + 4][y + 6][z] += 1

for i in range(n):
    states = odeint(makeLorentz(params[i]), state0, t)  # vector containing the (x,y,z) positions for each time step
    squares = np.int_(np.floor(states / 5))
    matrix = np.zeros((8, 12, 10))
    for elem in squares:
        square = tuple(elem)
        x, y, z = square
        matrix[x + 4][y + 6][z] += 1
    distancesL1[i] = L1(matrix0, matrix)
    distancesLinf[i] = Linf(matrix0, matrix)

print(distancesL1)
plt.plot(np.linspace(10, 5, 100), distancesL1, label="L1")
plt.plot(np.linspace(10, 5, 100), distancesLinf, label="Linf")
plt.xlabel("sigma")
plt.legend()
plt.grid()

plt.show()
"""
plt.title("Projection on the xy space")
plt.pcolormesh(x, y, xy.T / 5000)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.savefig("pdf_xy.png")
plt.show()

plt.title("Projection on the xz space")
plt.pcolormesh(x, z, xz.T / 5000)
plt.xlabel("x")
plt.ylabel("z")
plt.colorbar()
plt.savefig("pdf_xz.png")
plt.show()

plt.title("Projection on the yz space")
plt.pcolormesh(y, z, yz.T / 5000)
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar()
plt.savefig("pdf_yz.png")
plt.show()
"""

"""
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['True system'])
plt.draw()
plt.show()
"""

"""
PLOTLY : TRUE SYSTEM
"""

# Uncomment this section once you've installed the "Plotly" package

"""
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


fig = go.Figure(data=[go.Scatter3d(x=states[:, 0],y=states[:, 1],z=states[:, 2],
                                   mode='markers',
                                   marker=dict(
                                       size=2,
                                       opacity=0.8
    )                        
                                   )])
fig.update_layout(
    title='True system')
fig.update_scenes(aspectmode='data')
fig.show()
"""
