import matplotlib.pyplot as plt
import numpy as np

feature_list= [1,24,4]
score_list=[1,2,3]
x = np.array(feature_list)  # X-axis points
y = np.array(score_list)  # Y-axis points

plt.plot(x, y, label="a")  # Plot the chart
plt.plot(y,x, label="b")
plt.legend()
plt.show()
