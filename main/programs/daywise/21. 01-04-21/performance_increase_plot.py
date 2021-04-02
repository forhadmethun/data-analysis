import matplotlib.pyplot as plt
# line 1 points
x1 = [100,200,300,400,500]
y1 = [9.803445765590883,12.01164765833536,14.608104828925034,15.457413249211346,16.476583353554968]
# plotting the line 1 points
plt.plot(x1, y1, label = "Pool based")


# line 2 points
x2 = [100,200,300,400,500]
y2 = [0,2.026461229274839,3.215541785295601,3.2322893987606767,3.784960643108369]

# plotting the line 2 points
plt.plot(x2, y2, label = "Stream based ")
plt.xlabel('Samples')


# query y committee: line 3
x3 = [100,200,300,400,500]
y3 = [26.20037807183367,43.93194706994329,61.51228733459357,73.83742911153122,75.42533081285445]
# plotting the line 2 points
plt.plot(x3, y3, label = "Query by committee")
plt.xlabel('Samples')

# Set the y axis label of the current axis.
plt.ylabel('Performance Increase from base model')
# Set a title of the current axes.
plt.title('Performance Increase comparison on active learning')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
