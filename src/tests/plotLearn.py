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


fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
p1 = 0.5
p2 = 0.6
p3 = 0.1
alpha = -1
X = np.arange(-10, 10, 1.6)
Y = np.arange(-10, 10, 1.6)
X, Y = np.meshgrid(X, Y)
norm =(np.log(np.exp(X)+ np.exp(Y) +1))
normalpha =(np.log(np.exp(alpha*X)+ np.exp(alpha*Y) +1))/alpha
Z= np.log(p1*np.exp(X-norm)+ p2*np.exp(Y-norm)+p3*np.exp(-norm))- normalpha

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