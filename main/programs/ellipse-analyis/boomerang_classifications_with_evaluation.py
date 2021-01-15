from numpy import *
from matplotlib.pyplot import *
from math import sqrt
from math import pow
import random
import pandas as pd
import decimal
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
# x_number_list = []

# y axis value list.
# y_number_list = []



def draw_multiple_points(x_number_list, y_number_list):

    # Draw point based on above x, y axis values.
    scatter(x_number_list, y_number_list, s=20)

    # Set chart title.
    title("Extract Number Root ")

    # Set x, y label text.
    xlabel("Number")
    ylabel("Extract Root of Number")
    # plt.show()

def square(x):
    return x * x

def generate_boomerang():
    # x_point_list = []
    # y_point_list = []
    x_number_list = []
    y_number_list = []

    for m in range(-3, 2):
        for i in range(-25, 25):
            for j in range(0, random.randrange(3)):
                x = i + m
                y = square(i) + float(decimal.Decimal(random.randrange(-5, 6)))
                # x_point_list.append(x)
                # y_point_list.append(y)
                x_number_list.append(x)
                y_number_list.append(y)
                # if(i >= 0):
                #     first_class_list.append([x , y, 0])
                # else:
                #     first_class_list.append([x , y, 1])

    # plot(x_point_list, y_point_list)
    return x_number_list, y_number_list


if __name__ == '__main__':
    # draw ellipse
    # ellipse1, ellipse2 = draw_ellipse(0), draw_ellipse(50)

    # ellipse1 = generate_ellipse(0, 0, 80, 10, -80, 0, -45)
    # ellipse2 = generate_ellipse(0, 0, 80, 10, -80, 0, -135)


    x_number_list, y_number_list = generate_boomerang()

    first_class_list = []
    for i in range(0, len(x_number_list)):
        first_class_list.append([x_number_list[i], y_number_list[i], 0])



    draw_multiple_points(x_number_list, y_number_list)

    new_x_number_list = []
    new_y_number_list = []

    for i in range(0, len(x_number_list)):
        new_x_number_list.append(x_number_list[i] + 20)

    for i in range(0, len(y_number_list)):
        new_y_number_list.append(-1 * y_number_list[i])
        new_y_number_list[i] = new_y_number_list[i] + 720


    for i in range(0, len(new_x_number_list)):
        first_class_list.append([new_x_number_list[i], new_y_number_list[i], 1])

    draw_multiple_points(new_x_number_list,  new_y_number_list)

    # ellipse1 = generate_ellipse(0, 0, 80, 10, -80, 0, 0)
    # ellipse2 = generate_ellipse(0, 0, 80, 10, -80, 80, 0)

    # x_number_list.appefirst_class_listnd(ellipse1[2])
    # y_number_list.append(ellipse1[3])

    # x_number_list.append(ellipse2[2])
    # y_number_list.append(ellipse2[3])

    # first group
    # x_number_list = x_number_list + ellipse1[2]
    # y_number_list = y_number_list + ellipse1[3]
    #
    # first_class_list = []
    # for i in range(0, len(ellipse1[2])):
    #     first_class_list.append([ellipse1[2][i], ellipse1[3][i], 0])

    # print(first_class_list_np)


    # second group
    # x_number_list = x_number_list + ellipse2[2]
    # y_number_list = y_number_list + ellipse2[3]
    #
    # for i in range(0, len(ellipse2[2])):
    #     first_class_list.append([ellipse2[2][i], ellipse2[3][i], 1])

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

