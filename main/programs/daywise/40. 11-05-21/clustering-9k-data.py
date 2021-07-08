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

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan

from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, MeanShift
# from pyclustertend import hopkins
import warnings
from sklearn import datasets

warnings.filterwarnings('always')

# extracting 1k data of each category from the dataset
def normalize_datagram(df, n):
    df.sample(frac=1)
    header_attack_cat = df['label'].tolist()
    attack_categories = set(header_attack_cat)  # set of categories
    category_item_list = {}
    for category in attack_categories:
        category_item_list[category] = {
            "count": 0,
            "index": []
        }
    feature = 'label'
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

# https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/
# https://github.com/lachhebo/pyclustertend/blob/master/pyclustertend/hopkins.py
# https://pyclustertend.readthedocs.io/en/latest/
# https://pyclustertend.readthedocs.io/en/latest/#module-pyclustertend.hopkins
#
def hopkins(X):
    d = X.shape[1]
    # d = len(X) # columns
    n = len(X)  # rows
    m = int(0.1 * n)  # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(X)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print
        ujd, wjd
        H = 0

    return H

def compute_fisher_sorted(df):
    fisher_score = {}
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
        for i, (k, v) in enumerate(category_item_list.items()):
            uj = mean(v)
            oj = std(v)
            pj = len(v) / len(header_feature)
            num = num + pj * (uj - u) ** 2
            den = den + pj * oj ** 2
        F = num / den
        fisher_score[feature] = F
    fisher_score = {k: v for k, v in sorted(fisher_score.items(), key=lambda item: item[1], reverse=True)}
    sorted_features = list(fisher_score.keys())
    return sorted_features

# measure entropy of all features and return list of sorted feature
def compute_entropy_sorted(df):
    entropy_dict = {}
    for feature in df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', ]).columns:
        element_list = df[[feature]] #.tolist()
        # entropy_dict[feature] = entropy(element_list)
        entropy_dict[feature] = df[feature].tolist()
    entropy_dict = {k: v for k, v in sorted(entropy_dict.items(), key=lambda item: item[1])}
    sorted_features = list(entropy_dict.keys())
    return sorted_features
# {'dur': 0.9891754034113608, 'spkts': 0.9996907872122923, 'dpkts': 0.999966972875358, 'sbytes': 0.9995063123027357, 'dbytes': 0.9999680645096124, 'rate': 0.9999948552977795, 'sttl': 1.0, 'dttl': 1.0, 'sload': 0.9988602587111369, 'dload': 0.9997727114228457, 'sloss': 0.9986082535635271, 'dloss': 0.9999850706679022, 'sinpkt': 0.9994410839551949, 'dinpkt': 0.9999017753933076, 'sjit': 0.9991905858177081, 'djit': 0.9977035285443444, 'swin': 1.0, 'stcpb': 0.8564260937898603, 'dtcpb': 0.8570492405832978, 'dwin': 1.0, 'tcprtt': 0.9975274888847108, 'synack': 0.9981331018229241, 'ackdat': 0.9939733888618747, 'smean': 0.9879040536131436, 'dmean': 0.9914671179925086, 'trans_depth': 1.0, 'response_body_len': 0.9999614560201251, 'ct_srv_src': 1.0, 'ct_state_ttl': 1.0, 'ct_dst_ltm': 1.0, 'ct_src_dport_ltm': 1.0, 'ct_dst_sport_ltm': 1.0, 'ct_dst_src_ltm': 1.0, 'is_ftp_login': 1.0, 'ct_ftp_cmd': 1.0, 'ct_flw_http_mthd': 1.0, 'ct_src_ltm': 1.0, 'ct_srv_dst': 1.0, 'is_sm_ips_ports': 1.0, 'label': 1.0}
def compute_hopkins_sorted(df):
    hopkins_dict = {}
    for feature in df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', ]).columns:

        element_list = df[[feature]] #.tolist()
        X = datasets.load_iris().data
        hopkins_dict[feature] = hopkins(element_list)
    hopkins_dict = {k: v for k, v in sorted(hopkins_dict.items(), key=lambda item: item[1], reverse=True)}
    sorted_features = list(hopkins_dict.keys())
    return sorted_features

def kmean(df1, sorted_entropy_list ,feature_no):
    # apply knn algorithm
    mat = df1[sorted_entropy_list[1:feature_no]].values
    X = mat
    km =  sklearn.cluster.KMeans(n_clusters=11)
    km.fit(mat)
    labels = km.labels_
    results = pandas.DataFrame([dataset.index, labels]).T
    labels = km.labels_
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='rainbow')

    # category list
    list = ["Fuzzers", "Exploits", "Worms", "Shellcode", "Generic", "Analysis", "Backdoor", "DoS", "Reconnaissance",
            "Normal"]
    # print(list)
    dict = {}

    # populate dictionary where key is the category and value is the item count
    for j in range(0, 10):
        dict[j] = {"Fuzzers": 0, "Exploits": 0, "Worms": 0, "Shellcode": 0, "Generic": 0, "Analysis": 0,
                   "Backdoor": 0, "DoS": 0, "Reconnaissance": 0, "Normal": 0}

    # for each cluster
    for i in range (0, 10):
       # curerent cluster
       cur = df1[km.labels_ == i]
       # how many category item falls in the current cluster and increent their count based on the category!!
       for index, row in cur.iterrows():
            dict[i][list[int(row['label'])]] = dict[i][list[int(row['label'])]] + 1
    p = 0
    # d = {'x': 1, 'y': 2, 'z': 3}
    # for i, (key, value) in enumerate(d.items()):
    #     print(i, key, value)
    # w, h = 8, 5;
    # Matrix = [[0 for x in range(w)] for y in range(h)]
    l = 10
    res = [[0 for x in range(l + 1)] for y in range(l + 1)]
    for i, (key, value) in enumerate(dict.items()):
        q = 0
        for j, (key2, value2) in enumerate(value.items()):
            res[i + 1][j + 1] = str(value2)

    # print(res)
    # print(np.matrix(res))
    # print(json.dumps(dict, indent=2))
        # print(cur)
        # x = 0
    for i in range(0, 10):
        res[0][i + 1] = list[i]
        res[i + 1][0] = str(i)
    res[0][0] = ""
    data = res
    print("=========" + str(feature_no) + "=======")
    for r in data:
        print(r)
    print("================")

    # todo: saved the file here!!!
    # np.savetxt('fisher' + '-output-' + str(len(df1.columns)) + '-features' + '.csv',res,delimiter=",", fmt="%s")

if __name__ == "__main__":
    df_for_normalize = pd.read_csv('../../data/training.csv')
    df_normalized = normalize_datagram(df_for_normalize, 5000) #1k of each cat
    df1_back = df_normalized.copy(deep=True)
    sorted_entropy_list = compute_entropy_sorted(df1_back)
    # sorted_entropy_list = compute_fisher_sorted(df1_back)
    # sorted_entropy_list = compute_hopkins_sorted(df1_back)
    sorted_entropy_list.insert(0, 'label')
    dataset = df1_back
    dataset.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat'], axis=1, inplace=True)
    feature_list =  [3, 5, 10,15, 20] #[4, 5, 7] #[2,3,4,5, 6] # [5, 10, 20] # [2, 3, 5, 10, 20]
    for i in range(0, len(feature_list)):
        # print("# of Features: ")
        # print(feature_list[i])
        # print("Features: " )
        # print(sorted_entropy_list[:feature_list[i]])

        # print("Score: ")
        df1 = dataset[sorted_entropy_list[:feature_list[i]]]
        kmean(df1, sorted_entropy_list, feature_list[i])
        # print("-----------------")

plt.show()