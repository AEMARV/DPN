import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import os
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def sigmoid(lr):
    return 1/(1 + np.exp(-lr))

def problabel1(b1,b2):
    p1 = 1
    p2 = 1
    p3 = 1
    p4 = 0
    if b1 == -1:
        if b2== -1:
            return p1
        else:
            return p2
    else:
        if b2 == -1:
            return p3
        else:
            return p4


def calc_loss(l1,l2,alpha=1):


    loss = sigmoid(((-l1)+(-l2)))*problabel1(-1,-1) + (sigmoid(((l1)+(l2))))*(1- problabel1(-1,-1))
    loss += sigmoid(((l1) + (-l2))) * problabel1(1, -1) + (1 - sigmoid(((l1) + (-l2)))) * (1 - problabel1(1, -1))
    loss += sigmoid(((-l1) + (l2))) * problabel1(-1, 1) + (1 - sigmoid(((-l1) + (l2)))) * (1 - problabel1(-1, 1))
    loss += sigmoid(((l1) + (l2))) * problabel1(1, 1) + (1 - sigmoid(((l1) + (l2)))) * (1 - problabel1(1, 1))
    loss = np.log(loss)

    regloss = (sigmoid(-l1 - l2)**alpha + sigmoid(l1 + l2)**alpha)**(1/alpha)
    regloss += (sigmoid(-l1 + l2) ** alpha + sigmoid(l1 - l2) ** alpha) ** (1 / alpha)
    regloss += (sigmoid( l1 - l2) ** alpha + sigmoid(-l1 + l2) ** alpha) ** (1 / alpha)
    regloss += (sigmoid( l1 + l2) ** alpha + sigmoid(-l1 - l2) ** alpha) ** (1 / alpha)
    regloss = np.log(regloss)
    loss= loss - regloss
    return loss

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
p1 = 0.5
p2 = 0.6
p3 = 0.1
alpha = -1
l1 = np.arange(-10, 10, 0.2)
l2 = np.arange(-10, 10, 0.2)
X, Y = np.meshgrid(l1, l2)
Z= calc_loss(X,Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-10, 0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()