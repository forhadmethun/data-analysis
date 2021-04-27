import collections
import json

import pandas
import pandas as pd
import sklearn
from matplotlib import cm
from numpy import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, silhouette_score, silhouette_samples

from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, MeanShift
from scipy.spatial.distance import pdist, squareform

import warnings

warnings.filterwarnings('always')

# extracting 1k data of each category from the dataset
def normalize_datagram(df, n):
    df.sample(frac=1)
    header_attack_cat = df['attack_cat'].tolist()
    attack_categories = set(header_attack_cat)  # set of categories
    category_item_list = {}
    for category in attack_categories:
        category_item_list[category] = {
            "count": 0,
            "index": []
        }
    feature = 'attack_cat'
    header_feature = df[feature].tolist()
    removal_list = []
    for index, val in enumerate(header_feature):
        if (category_item_list[header_attack_cat[index]]['count'] > n):
            removal_list.append(index)
        category_item_list[header_attack_cat[index]]['count'] = \
            category_item_list[header_attack_cat[index]]['count'] + 1
    df.drop(df.index[removal_list], inplace=True)
    list = ["Fuzzers", "Exploits", "Worms", "Shellcode", "Generic", "Analysis", "Backdoor", "DoS", "Reconnaissance",
            "Normal"]
    # for index, val in enumerate(header_feature):
    for i, l in enumerate(list):
        df.loc[df['attack_cat'] == l, ['label']] = i
    return df

# measure entropy for a feature from the list of data
def entropy(labels):
    s = []
    maximum = max(labels)
    minimum = min(labels)
    width = maximum - minimum
    per_fraction = width / len(labels)
    dict = {}
    for i, val in enumerate(labels):
        index = math.floor((val - minimum) / per_fraction)
        s.append(index)
        if index not in dict:
            dict[index] = 0
        else:
            dict[index] = dict[index] + 1
    probabilities = [n_x/len(s) for x,n_x in collections.Counter(s).items()]
    e_x = [-(p_x*math.log(p_x,2) + (1 - p_x)*math.log(1 - p_x,2)) for p_x in probabilities]
    entropy = sum(e_x)
    return entropy

# measure entropy of all features and return list of sorted feature
def compute_entropy_sorted(df):
    entropy_dict = {}
    for feature in df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', ]).columns:
        element_list = df[feature].tolist()
        entropy_dict[feature] = entropy(element_list)
    entropy_dict = {k: v for k, v in sorted(entropy_dict.items(), key=lambda item: item[1])}
    # print("=================================")
    # print("----- ENTROPY OF FEATURES ------")
    # print("=================================")
    # print(json.dumps(entropy_dict, indent=2))
    for f in entropy_dict:
        # print(f)
        print("%.5f" % entropy_dict[f])

    sorted_features = list(entropy_dict.keys())
    return sorted_features

def showBar(data):
    length = len(data)
    X = np.arange(length)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    p = 0
    for i in range(0, length):
        ax.bar(X + p, data[i], width=1 / (length + 1))
        p = p + 1 / (length + 5)
    ax.set_ylabel('distance')
    ax.set_title('Amount Frequency')
    ax.set_xlabel('Amount ($)')
    plt.show()

def kmean(df1):
    mat = df1.values
    X = mat
    km =  sklearn.cluster.KMeans(n_clusters=10)
    km.fit(mat)
    labels = km.labels_
    results = pandas.DataFrame([dataset.index, labels]).T
    labels = km.labels_
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='rainbow')

    list = ["Fuzzers", "Exploits", "Worms", "Shellcode", "Generic", "Analysis", "Backdoor", "DoS", "Reconnaissance",
            "Normal"]

    dict = {}

    for j in range(0, 10):
        dict[j] = {"Fuzzers": 0, "Exploits": 0, "Worms": 0, "Shellcode": 0, "Generic": 0, "Analysis": 0,
                   "Backdoor": 0, "DoS": 0, "Reconnaissance": 0, "Normal": 0}

    clusters = []
    for i in range (0, 10):
       cur = df1[km.labels_ == i]
       each_cluster_mat = []
       for index, row in cur.iterrows():
           each_cluster_mat.append(row)
           dict[i][list[row['label']]] = dict[i][list[row['label']]] + 1
       clusters.append(each_cluster_mat)
    print(json.dumps(dict, indent=2))

    # distance

    first = clusters[0]
    second = clusters[1]
    parent_dist = 0

    # for i in range(0, int(len(first)/5)):
    #     dist = 0
    #     for j in range(0, int(len(second)/5)):
    #         dist += np.linalg.norm(first[i] - second[j])
    #     dist = dist / (len(second)/5)
    #     parent_dist += dist
    #     x = 0
    # parent_dist = parent_dist / (len(first)/5)

    l = 9
    ans = [[0 for x in range(l + 1)] for y in range(l + 1)]

    # ans = [[]]
    for m in range(0, len(clusters)):
        for n in range(m + 1, len(clusters)):
            first = clusters[m]
            second = clusters[n]
            parent_dist = 0

            for i in range(0, int(len(first) / 50)):
                dist = 0
                for j in range(0, int(len(second) / 50)):
                    dist += np.linalg.norm(first[i] - second[j])
                dist = dist / (len(second) / 50)
                parent_dist += dist
                x = 0
            parent_dist = parent_dist / (len(first) / 50)
            ans[m][n] = round(parent_dist, 2)
            ans[n][m] = round(parent_dist, 2)
        k = 0

    data = ans
    print(ans)
    showBar(ans)
    # X = np.arange(4)
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # for i in range(0, 10):
    #     ax.bar(X + 0.00 + i * .05, data[i], color='b', width=0.10)
        # ax.bar(X + 0.25, data[1], color='g', width=0.25)
        # ax.bar(X + 0.50, data[2], color='r', width=0.25)
    # plt.show()
    y = 0

    # np.linalg.norm(clusters[0][0].values.flatten() - clusters[0][1].values.flatten())


    # distances = pdist(results, metric='euclidean')
    # dist_matrix = squareform(distances)
    # print(dist_matrix)
        # print(cur)
        # x = 0



if __name__ == "__main__":
    df_for_normalize = pd.read_csv('../../data/training.csv')
    df_normalized = normalize_datagram(df_for_normalize, 1000) #1k of each cat
    df1_back = df_normalized.copy(deep=True)
    sorted_entropy_list = compute_entropy_sorted(df1_back)

        # print("-----------------")

plt.show()