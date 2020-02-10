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

def lognorm(l1,l2):
	l1 = l1 - np.logaddexp(l1,l2)
	return l1



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
Z= np.exp(lognorm(X,Y))

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