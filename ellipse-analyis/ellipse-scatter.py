import numpy
import matplotlib.pyplot as plt
import random

# ref : https://stackoverflow.com/a/46840451/3173062

# Returns theta in [-pi/2, 3pi/2]
def generate_theta(a, b):
    u = random.random() / 4.0
    theta = numpy.arctan(b/a * numpy.tan(2*numpy.pi*u))

    v = random.random()
    if v < 0.25:
        return theta
    elif v < 0.5:
        return numpy.pi - theta
    elif v < 0.75:
        return numpy.pi + theta
    else:
        return -theta

def radius(a, b, theta):
    return a * b / numpy.sqrt((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)

def random_point(a, b):
    random_theta = generate_theta(a, b)
    max_radius = radius(a, b, random_theta)
    random_radius = max_radius * numpy.sqrt(random.random())

    return numpy.array([
        random_radius * numpy.cos(random_theta),
        random_radius * numpy.sin(random_theta)
    ])

a = 2
b = 1

points = numpy.array([random_point(a, b) for _ in range(800)])
points2 = numpy.array([random_point(a, b) for _ in range(800)])

plt.scatter(points[:,0], [i + 2.5 for i in points[:,1]])
plt.scatter(points2[:,0], points2[:,1])
plt.show()