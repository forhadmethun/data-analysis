from numpy import *
from matplotlib.pyplot import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report

import numpy
import matplotlib.pyplot as plt
import random
import time

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




if __name__ == '__main__':
    count = 1

    x = 10
    while x != -6:
        print("------ CASE : " + str(count) + "-----------\n")
        count = count + 1
        x_axis_point_list_of_bottom_ellipse = points2[:, 0]
        y_axis_point_list_of_bottom_ellipse = points2[:, 1]
        plt.scatter(x_axis_point_list_of_bottom_ellipse, y_axis_point_list_of_bottom_ellipse)

        x_axis_point_list_of_moving_ellipse = points[:, 0]
        x_axis_point_list_of_moving_ellipse = [i + random.uniform(0, 1) for i in x_axis_point_list_of_moving_ellipse]
        y_axis_point_list_of_moving_ellipse = [i + x + random.uniform(0, 1) for i in points[:, 1]]
        plt.scatter(x_axis_point_list_of_moving_ellipse, y_axis_point_list_of_moving_ellipse)

        plt.show()
        x = x - 2
        # time.sleep(1)

        first_class_list = []
        first_class_list.clear()
        for i in range(0, len(x_axis_point_list_of_bottom_ellipse)):
            first_class_list.append([x_axis_point_list_of_bottom_ellipse[i], y_axis_point_list_of_bottom_ellipse[i], 0])

        for i in range(0, len(x_axis_point_list_of_moving_ellipse)):
            first_class_list.append([x_axis_point_list_of_moving_ellipse[i], y_axis_point_list_of_moving_ellipse[i], 1])


        first_class_list_np = array(first_class_list)
        df = pd.DataFrame({'Column1': first_class_list_np[:, 0], 'Column2': first_class_list_np[:, 1], 'Column3': first_class_list_np[:, 2]})
        scaler = StandardScaler()
        scaler.fit(df.drop('Column3', axis=1))
        sc_transform = scaler.transform(df.drop('Column3', axis=1))
        sc_df = pd.DataFrame(sc_transform)
        X = sc_transform
        y = df['Column3']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        error_rates = []
        for a in range(1, 40):
            k = a
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            preds = knn.predict(X_test)
            error_rates.append(np.mean(y_test - preds))
        k = 22
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))
