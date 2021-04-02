import collections
import math
import warnings
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pandas
import pandas as pd
import sklearn
from modAL.uncertainty import classifier_uncertainty
from numpy import *
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee

warnings.filterwarnings('always')

# extracting 1k data of each category from the dataset
category_list = ["Fuzzers", "Exploits", "Worms", "Shellcode", "Generic", "Analysis", "Backdoor", "DoS",
                 "Reconnaissance",
                 "Normal"]

def normalize_datagram(df, n):
    df.sample(frac=1)
    header_attack_cat = df['attack_cat'].tolist()
    attack_categories = set(header_attack_cat)  # set of categories
    category_item_list = {}
    normalized_category_item_list = {}
    for category in attack_categories:
        category_item_list[category] = {
            "count": 0,
            "index": []
        }
        normalized_category_item_list[category] = {
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
        # category_item_list[header_attack_cat[index]]['index'].append(index)
    df.drop(df.index[removal_list], inplace=True)


    # normalized calculation after removing the extra data
    header_feature = df[feature].tolist()
    header_attack_cat = df['attack_cat'].tolist()
    for index, val in enumerate(header_feature):
        normalized_category_item_list[header_attack_cat[index]]['count'] = \
            normalized_category_item_list[header_attack_cat[index]]['count'] + 1
        normalized_category_item_list[header_attack_cat[index]]['index'].append(index)

    first_item_index_of_each_category = []
    for key, value in normalized_category_item_list.items():
        for i in range(0, 100):
            first_item_index_of_each_category.append(value['index'][i])



    # for index, val in enumerate(header_feature):
    for i, l in enumerate(category_list):
        df.loc[df['attack_cat'] == l, ['label']] = i
    return df, first_item_index_of_each_category

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

    k = 0
    # intra_cluster_distance(clusters)
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

def percentage_increase(a):
    for i in range(1, len(a)):
        # per = (a[i] - a[i - 1]) / a[i - 1]
        per = (a[i] - a[0]) / a[0]
        print(per * 100)

def active_learn(df1, first_item_index_of_each_category):
    train_idx = first_item_index_of_each_category
    # X_train = iris['data'][train_idx]
    # y_train = iris['target'][train_idx]

    # initial training data
    data = df1.values[:,1:]
    target = df1['label'].values

    X_full = df1.values[:, 1:]
    y_full = df1['label'].values


    X_train = df1.values[:,1:][train_idx] #item from second column as the first column is the label..
    y_train = df1['label'].values[train_idx]

    # X_pool = np.delete(data, train_idx, axis=0)
    # y_pool = np.delete(target, train_idx)

    X_pool = deepcopy(X_full)
    y_pool = deepcopy(y_full)

    # initializing Committee members
    n_members = 2
    learner_list = list()

    for member_idx in range(n_members):
        # initial training data
        n_initial = 5
        train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
        X_train = X_pool[train_idx]
        y_train = y_pool[train_idx]

        # creating a reduced copy of the data with the known instances removed
        X_pool = np.delete(X_pool, train_idx, axis=0)
        y_pool = np.delete(y_pool, train_idx)

        # initializing learner
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=X_train, y_training=y_train
        )
        learner_list.append(learner)
        # assembling the committee
    committee = Committee(learner_list=learner_list)

    print('Committee initial predictions, accuracy = %1.3f' % committee.score(data, target))

    performance_array = []
    n_queries = 505
    for idx in range(n_queries):
        query_idx, query_instance = committee.query(X_pool)
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
        learner_score = committee.score(data, target)
        print('Committee %d th query predictions, accuracy = %1.3f' % (idx , learner_score))
        if (idx % 100 == 0):
            performance_array.append(learner_score)
    percentage_increase(performance_array)

        ###


    # initializing the active learner
    # learner = ActiveLearner(
    #     estimator=RandomForestClassifier(),
    #     X_training=X_train, y_training=y_train
    # )

    # print('Initial prediction accuracy: %f' % learner.score(X_full, y_full))
    # index = 0
    # performance_array = []
    ## learning until the accuracy reaches a given threshold
    # while learner.score(X_full, y_full) < 0.90:
    #     stream_idx = np.random.choice(range(len(X_full)))
    #     if classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1)) >= 0.4:
    #         learner.teach(X_full[stream_idx].reshape(1, -1), y_full[stream_idx].reshape(-1, ))
    #         learner_score = learner.score(X_full, y_full)
    #         print('Item no. %d queried, new accuracy: %f' % (stream_idx, learner_score))
    #         index = index + 1
    #         if index == 505:
    #             break
    #         if (index % 100 == 0):
    #             performance_array.append(learner_score)
    # percentage_increase(performance_array)

if __name__ == "__main__":
    df_for_normalize = pd.read_csv('../../data/training.csv')
    df_normalized, first_item_index_of_each_category = normalize_datagram(df_for_normalize, 1000) #1k of each cat
    df1_back = df_normalized.copy(deep=True)
    sorted_entropy_list = compute_entropy_sorted(df1_back)
    sorted_entropy_list.insert(0, 'label')
    dataset = df1_back
    dataset.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat'], axis=1, inplace=True)
    feature_list = [20] #[2,3,4,5, 6] # [5, 10, 20] # [2, 3, 5, 10, 20]
    for i in range(0, len(feature_list)):
        df1 = dataset[sorted_entropy_list[:feature_list[i]]]
        # kmeanAndFindDistances(df1)
        active_learn(df1, first_item_index_of_each_category)

plt.show()