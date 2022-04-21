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

"""
Fonction plutot utile partout
"""

def makeLorentz(params):
    def Lorenz(state, t):
        x, y, z = state  # Unpack the state vector
        return params[0] * (y - x), x * (params[1] - z) - y, x * y - params[2] * z  # Derivatives

    return Lorenz

"""

Partie 1.1.1

"""

params = [10.0, 28.0, 8.0 / 3.0]
state0 = [1.0, 1.0, 1.0]  # initial condition
t = np.arange(0.0, 100.0, 0.02)  # time vector
states = odeint(makeLorentz(params), state0, t)  # vector containing the (x,y,z) positions for each time step
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

Partie 1.1.2

"""

def L1(X, Y):
    distance = abs(X - Y)
    return np.sum(distance)


def Linf(X, Y):
    distance = abs(X - Y)
    return np.max(distance)


"""

partie 1.1.3

"""

n = 100
distancesL1 = np.zeros(n)
distancesLinf = np.zeros(n)

params = np.zeros((n, 3))
params[:, 0] = np.linspace(10, 5, n)
params[:, 1] = np.linspace(28, 28, n)
params[:, 2] = np.linspace(8/3, 8/3, n)
state0 = [1, 1, 1]

t = np.arange(0.0, 100.0, 0.02)  # time vector
states = odeint(makeLorentz(params[0]), state0, t)  # vector containing the (x,y,z) positions for each time step
squares = np.int_(np.floor(states / 5))
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

plt.plot(np.linspace(10, 5, n), distancesL1, label="L1")
plt.plot(np.linspace(10, 5, n), distancesLinf, label="Linf")
plt.xlabel("sigma")
plt.ylabel("distance")
plt.title("distance between the two statistical distributions")
plt.legend()
plt.grid()
plt.show()

"""

partie 1.1.4

"""
n = 100
distancesL1 = np.zeros(n)
distancesLinf = np.zeros(n)

params = [10.0, 28.0, 8.0 / 3.0]
states0 = np.zeros((n, 3))
states0[:, 0] = np.linspace(1, 10, n)
states0[:, 1] = np.linspace(1, 10, n)
states0[:, 2] = np.linspace(1, 10, n)

t = np.arange(0.0, 100.0, 0.02)  # time vector
states = odeint(makeLorentz(params), states0[0], t)  # vector containing the (x,y,z) positions for each time step
squares = np.int_(np.floor(states / 5))
matrix0 = np.zeros((8, 12, 10))
for elem in squares:
    square = tuple(elem)
    x, y, z = square
    matrix0[x + 4][y + 6][z] += 1

for i in range(n):
    statesi = odeint(makeLorentz(params), states0[i], t)  # vector containing the (x,y,z) positions for each time step
    squares = np.int_(np.floor(statesi / 5))
    matrix = np.zeros((8, 12, 10))
    for elem in squares:
        square = tuple(elem)
        x, y, z = square
        matrix[x + 4][y + 6][z] += 1
    distancesL1[i] = L1(matrix0, matrix)
    distancesLinf[i] = Linf(matrix0, matrix)

plt.plot(np.linspace(1, 10, n), distancesL1, label="L1")
plt.plot(np.linspace(1, 10, n), distancesLinf, label="Linf")
plt.xlabel("initial state coordinates")
plt.ylabel("distance")
plt.title("distance between the two statistical distributions")
plt.legend()
plt.grid()
plt.show()