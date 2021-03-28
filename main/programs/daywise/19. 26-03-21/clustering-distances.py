import collections
import json
import math

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

def kmeanAndFindDistances(df1):
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
    intra_cluster_distance(clusters)
    # inter_cluster_distance(clusters)

def intra_cluster_distance(clusters):
    for m in range(0, len(clusters)):
        first = clusters[m]
        cluster_intra_distances = []
        for i in range(0, int(len(first)/5)):
            for j in range(i + 1, int(len(first)/5)):
                cluster_intra_distances.append(np.linalg.norm(first[i] - first[j]))
        plt.hist(cluster_intra_distances, density=True, bins=30)  # density=False would make counts
        plt.xlabel('Distance measured')
        plt.ylabel('Freq.')
        plt.show()

def inter_cluster_distance(clusters):
    for m in range(0, len(clusters)):
        cluster_inter_distance = []
        for n in range(m + 1, len(clusters)):
            first = clusters[m]
            second = clusters[n]
            for i in range(0, int(len(first) / 50)):
                for j in range(0, int(len(second) / 50)):
                    cluster_inter_distance.append(np.linalg.norm(first[i] - second[j]))
        plt.hist(cluster_inter_distance, density=True, bins=30)  # density=False would make counts
        plt.xlabel('Distance measured')
        plt.ylabel('Freq.')
        plt.show()

if __name__ == "__main__":
    df_for_normalize = pd.read_csv('../../data/training.csv')
    df_normalized = normalize_datagram(df_for_normalize, 1000) #1k of each cat
    df1_back = df_normalized.copy(deep=True)
    sorted_entropy_list = compute_entropy_sorted(df1_back)
    sorted_entropy_list.insert(0, 'label')
    dataset = df1_back
    dataset.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat'], axis=1, inplace=True)
    feature_list = [5] #[2,3,4,5, 6] # [5, 10, 20] # [2, 3, 5, 10, 20]
    for i in range(0, len(feature_list)):
        df1 = dataset[sorted_entropy_list[:feature_list[i]]]
        kmeanAndFindDistances(df1)

plt.show()