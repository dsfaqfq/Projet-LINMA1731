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

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


# Lorenz model

def Lorenz(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives


state0 = [1.0, 1.0, 1.0]  # initial condition
t = np.arange(0.0, 100.0, 0.02)  # time vector

states = odeint(Lorenz, state0, t)  # vector containing the (x,y,z) positions for each time step

squares = np.int_(np.floor(states / 5))

matrix = np.zeros((8, 12, 10))
for elem in squares:
    square = tuple(elem)
    x, y, z = square
    matrix[x + 4][y + 6][z] += 1

xy = np.sum(matrix, axis=2)
yz = np.sum(matrix, axis=0)
xz = np.sum(matrix, axis=1)

x, y, z = np.linspace(-20, 20, 9), np.linspace(-30, 30, 13), np.linspace(0, 50, 11)

plt.pcolormesh(x, y, xy.T)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()

plt.pcolormesh(x, z, xz.T)
plt.xlabel("x")
plt.ylabel("z")
plt.colorbar()
plt.show()

plt.pcolormesh(y, z, yz.T)
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['True system'])
plt.draw()
plt.show()

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
