import pandas
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from numpy import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score

from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings('always')
# def calculate_entropy_classification(list_j, radius, interval_step, interval_m):
#     entropy_j = 0
#     for i in range(1, interval_m + 1):
#         start_interval, end_interval = -radius + interval_step * (i - 1), -radius + interval_step * i
#         pj = get_number_of_points_in_interval(list_j, start_interval, end_interval) / len(list_j)
#         if pj != 0:
#             entropy_j = entropy_j + pj * log2(pj)
#     return entropy_j

def measure_entropy(p):
   return - p*np.log2(p) - (1 - p)*np.log2((1 - p))


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
        category_item_list[header_attack_cat[index]]['count'] = category_item_list[header_attack_cat[index]][
                                                                    'count'] + 1
    df.drop(df.index[removal_list], inplace=True)
    return df


def compute_entropy_sorted(df):
    entropy = {}
    for feature in df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', 'label']).columns:
        header_attack_cat = df['attack_cat'].tolist()
        attack_categories = set(header_attack_cat)
        category_item_list = {}
        for category in attack_categories:
            category_item_list[category] = []
        header_feature = df[feature].tolist()
        for index, val in enumerate(header_feature):
            category_item_list[header_attack_cat[index]].append(val)
        u = mean(header_feature)
        F = 0
        num = 0
        den = 0
        # todo: need to write the code related to forumula of the entropy!!
        # for i, (k, v) in enumerate(category_item_list.items()):
            # uj = mean(v)
            # oj = std(v)
            # pj = mean(v)
            # num = num + measure_entropy(pj)
            # den = den + pj * oj ** 2

        entropy[feature] = num
    entropy = {k: v for k, v in sorted(entropy.items(), key=lambda item: item[1])}
    # print(json.dumps(fisher_score, indent=2))
    sorted_features = list(entropy.keys())
    return sorted_features

if __name__ == "__main__":
    df = pd.read_csv('../../data/training.csv')
    df_for_normalize = pd.read_csv('../../data/training.csv')
    df_normalized = normalize_datagram(df_for_normalize, 1000)
    df_back = df_normalized.copy(deep=True)
    df1_back = df_back.copy(deep=True)
    entropy_list = compute_entropy_sorted(df1_back)
    dataset = df1_back
    dataset.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', 'label'], axis=1, inplace=True)
    mat = dataset.values
    X = mat
    # plt.scatter(X[:, 0], X[:, 1], label='True Position')
    km = sklearn.cluster.KMeans(n_clusters=10)
    km.fit(mat)
    labels = km.labels_
    results = pandas.DataFrame([dataset.index, labels]).T
    labels = km.labels_
    score = sklearn.metrics.silhouette_score(mat, labels, metric='euclidean')
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='rainbow')
    print(score)

plt.show()