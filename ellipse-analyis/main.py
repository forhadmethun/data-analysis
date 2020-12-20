from numpy import *
from matplotlib.pyplot import *

# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def fisher(x, y):
    u = mean(x + y)

    # for y axis
    uy = mean(y)
    oy = std(y)
    num_y = 0
    denom_y = 0
    for py in y:
        num_y = num_y + py * pow((uy - u), 2)
        denom_y = denom_y + py * pow(oy, 2)
    fy = num_y / denom_y

    # for x axis
    ux = mean(x)
    ox = std(x)
    num_x = 0
    denom_x = 0
    for px in x:
        num_x = num_x + px * pow((ux - u), 2)
        denom_x = denom_x + px * pow(ox, 2)
    fx = num_x / denom_x

    return fx, fy


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
