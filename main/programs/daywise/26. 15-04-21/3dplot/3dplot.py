import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
#
# def grid(x, y, z, resX=100, resY=100):
#     "Convert 3 column data to matplotlib grid"
#     xi = np.linspace(min(x), max(x), resX)
#     yi = np.linspace(min(y), max(y), resY)
#     Z = griddata(x, y, z, xi, yi)
#     X, Y = np.meshgrid(xi, yi)
#     return X, Y, Z

x,y,z = np.loadtxt("data.txt", unpack=True)
# # plt.tricontour(x, y, z, 15, linewidths=0.5, colors='k')
# # plt.tricontourf(x, y, z, 15)
# X, Y, Z = grid(x, y, z)
# plt.contourf(X, Y, Z)
# plt.plot()


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(x, y, z, 'red')

ax.view_init(60, 50)
plt.show()