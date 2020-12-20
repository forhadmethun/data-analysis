from numpy import *
from matplotlib.pyplot import *
from math import sqrt
from math import pow
import random
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report

NUM_OF_POINTS = 360

CENTER_X = 0
CENTER_Y = 0

RADIUS_ON_X_AXIS = 80
RADIUS_ON_Y_AXIS = 10

FIRST_ELLIPSE_SHIFT_Y = 0
SECOND_ELLIPSE_SHIFT_Y = 0
SECOND_ELLIPSE_SHIFT_Y_PREV = 50

# x axis value list.
x_number_list = []

# y axis value list.
y_number_list = []



def draw_multiple_points():


    # Draw point based on above x, y axis values.
    scatter(x_number_list, y_number_list, s=20)

    # Set chart title.
    title("Extract Number Root ")

    # Set x, y label text.
    xlabel("Number")
    ylabel("Extract Root of Number")
    # plt.show()

def generate_ellipse(center_x, center_y, radius_x, radius_y, shift_x, shift_y, rotate_degree):
    global NUM_OF_POINTS
    theta = radians(rotate_degree)

    t = linspace(0, NUM_OF_POINTS, NUM_OF_POINTS)


    x = radius_x * cos(radians(t)) + center_x + shift_x
    y = radius_y * sin(radians(t)) + center_y + shift_y

    x_point_list = []
    y_point_list = []

    u = x * cos(theta) - y * sin(theta)
    v = x * sin(theta) + y * cos(theta)
    # u = x
    # v = y

    h = center_x #+ shift_x
    k = center_y #+ shift_y

    # h = h * cos(theta) - h * sin(theta)
    # k = k * sin(theta) + k * cos(theta)


    for i in range(1, 1000):
        sq_radius_x = radius_x * radius_x
        sq_radius_y = radius_y * radius_y

        rx = random.randrange(-100, 100)
        ry = random.randrange(0, 100)

        # rx = rx * cos(theta) - ry * sin(theta)
        # ry = rx * sin(theta) + ry * cos(theta)



        # x_h = rx - h
        # y_h = ry - k

        x_h = (rx - h) * cos(theta) + (ry-k) * sin(theta)
        y_h =  (rx - h) * sin(theta) + (ry-k) * cos(theta)


        sq_x_h = x_h * x_h
        sq_y_h = y_h * y_h
        if ((sq_x_h / sq_radius_x) + (sq_y_h / sq_radius_y) <= 1):
            x_point_list.append(rx)
            y_point_list.append(ry)
    plot(u, v)
    return u, v, x_point_list, y_point_list


if __name__ == '__main__':
    # draw ellipse
    # ellipse1, ellipse2 = draw_ellipse(0), draw_ellipse(50)

    ellipse1 = generate_ellipse(0, 0, 80, 10, -80, 0, -45)
    ellipse2 = generate_ellipse(0, 0, 80, 10, -80, 0, -135)

    # ellipse1 = generate_ellipse(0, 0, 80, 10, -80, 0, 0)
    # ellipse2 = generate_ellipse(0, 0, 80, 10, -80, 80, 0)

    # x_number_list.append(ellipse1[2])
    # y_number_list.append(ellipse1[3])

    # x_number_list.append(ellipse2[2])
    # y_number_list.append(ellipse2[3])

    # first group
    x_number_list = x_number_list + ellipse1[2]
    y_number_list = y_number_list + ellipse1[3]

    first_class_list = []
    for i in range(0, len(ellipse1[2])):
        first_class_list.append([ellipse1[2][i], ellipse1[3][i], 0])

    # print(first_class_list_np)


    # second group
    x_number_list = x_number_list + ellipse2[2]
    y_number_list = y_number_list + ellipse2[3]

    for i in range(0, len(ellipse2[2])):
        first_class_list.append([ellipse2[2][i], ellipse2[3][i], 1])

    first_class_list_np = array(first_class_list)

    df = pd.DataFrame({'Column1': first_class_list_np[:, 0], 'Column2': first_class_list_np[:, 1], 'Column3': first_class_list_np[:, 2]})
    print(df)
    # pd.read_csv()
    # df.head()



    scaler = StandardScaler()

    scaler.fit(df.drop('Column3', axis=1))
    sc_transform = scaler.transform(df.drop('Column3', axis=1))
    sc_df = pd.DataFrame(sc_transform)
    print(sc_df)

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
    # figure(figsize=(10, 7))
    # plot(range(1, 40), error_rates, color='blue', linestyle='dashed', marker='o',
    #          markerfacecolor='red', markersize=10)
    # title('Error Rate vs. K Value')
    # xlabel('K')
    # ylabel('Error Rate')

    draw_multiple_points()
    k = 22
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    show()

    # Fisher score
    # while True:
    #     interval_m = int(input("Enter number of intervals: "))

