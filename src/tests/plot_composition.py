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

def calc_loss(l1,l2,alpha=10):
    p10 = sigmoid(-l1)
    p11 = sigmoid(l1)


    loss0 = np.log((sigmoid(-l1)*sigmoid(l2) + sigmoid(l1)*sigmoid(-l2)))
    loss1 = np.log(sigmoid(l1)*(sigmoid(l2*2)) + sigmoid(-l1)*(sigmoid(-l2*2)))
    loss = loss0 + loss1

    regloss= (sigmoid(-l1)*(sigmoid(l2)))**alpha
    regloss += (sigmoid(-l1)*(sigmoid(-l2)))**alpha
    regloss += (sigmoid(l1) * (sigmoid(l2)))**alpha
    regloss += (sigmoid(l1) * (sigmoid(-l2)))**alpha
    regloss = np.log(regloss ** (1/alpha))
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
norm =(np.log(np.exp(X)+ np.exp(Y) +1))
normalpha =(np.log(np.exp(alpha*X)+ np.exp(alpha*Y) +1))/alpha
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