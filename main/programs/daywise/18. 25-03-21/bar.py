import numpy as np
import matplotlib.pyplot as plt


# ax.bar(X + 0.00, data[0],  width = 0.25)
# ax.bar(X + 0.25, data[1],  width = 0.25)
# ax.bar(X + 0.50, data[2],  width = 0.25)
# ax.bar(X + 0.70, data[3],  width = 0.10)

# X = np.arange(5)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# length = len(data)
# p = 0
# for i in range(0, length):
#     ax.bar(X + p, data[i], width=1/(length+1))
#     p = p + 1/(length+1)
#

def showBar(data):
    length = len(data)
    X = np.arange(length)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_title('axes title', color='brown')
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
    # ax.text(3, 8, 'boxed italics text in data coords', style='italic',
    #         bbox={'facecolor': 'red'})
    # ax.text(2, 6, r'an equation: $E = mc^2$', fontsize=15)
    # ax.text(4, 0.05, 'colored text in axes coords',
    #         verticalalignment='bottom', color='green', fontsize=15)
    # ax.plot([2], [1], 'o')
    # ax = freq_series.plot(kind='bar')
    # ax.set_title('Amount Frequency')
    # ax.set_xlabel('Amount ($)')
    # ax.set_ylabel('Frequency')
    # x_labels = [0, 1, 2, 3]
    # ax.set_xticklabels(x_labels)
    # plt.xlabel('Number')
    # plt.ylabel("Square")

    p = 0
    for i in range(0, length):
        ax.bar(X + p, data[i], width=1 / (length + 1))
        p = p + 1 / (length + 1)
    # keys = [0, 1, 2, 3 ,4]
    # ax.set_xticklabels(keys)
    # ax.set_xticks(np.arange(len(keys)))
    ax.set_title("sine wave")
    ax.set_xlabel('angle')
    ax.set_ylabel('sine')


    plt.xlabel("angle")
    plt.ylabel("sine")
    plt.title('sine wave')
    plt.show()

def testBar():
    data = [[30, 25, 50, 20, 21, 12],
            [40, 23, 51, 17, 21, 22],
            [85, 22, 45, 19, 11, 22],
            [75, 22, 15, 19, 21, 28],
            [75, 22, 15, 19, 21,5],
            [75, 22, 15, 19, 21,5]
            ]
    showBar(data)
testBar()
