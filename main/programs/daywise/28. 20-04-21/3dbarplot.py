from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# xpos = [1,2,3,4,5,6,7,8,9,10]
# ypos = [2,3,4,5,1,6,2,1,7,2]
# num_elements = len(xpos)
# zpos = [0,0,0,0,0,0,0,0,0,0]
# dx = np.ones(10)
# dy = np.ones(10)
# dz = [1,2,3,4,5,6,7,8,9,10]

x = [0, 2, 3]  # x coordinates of each bar
y = [3,4, 5]  # y coordinates of each bar
z = [0, 0, 0]  # z coordinates of each bar
dx = [1, 1, 1]  # Width of each bar
dy = [1, 1, 1]  # Depth of each bar
dz = [0.5, 0.4, 0.7]        # Height of each bar


# ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
ax1.bar3d(x, y, z, dx, dy, dz, color='#00ceaa')
plt.show()