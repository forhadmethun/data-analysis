import collections
import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import sklearn
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_uncertainty
from numpy import *
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
        for i in range(0, 155):
            if (i< len(value['index'])):
             first_item_index_of_each_category.append(value['index'][i])

    # for index, val in enumerate(header_feature):
    for i, l in enumerate(category_list):
        df.loc[df['attack_cat'] == l, ['label']] = i
    return df, first_item_index_of_each_category

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

def percentage_increase(a):
    for i in range(1, len(a)):
        # per = (a[i] - a[i - 1]) / a[i - 1]
        per = (a[i] - a[0]) / a[0]
        print(per * 100)

def percentage_increase(a):
    for i in range(1, len(a)):
        # per = (a[i] - a[i - 1]) / a[i - 1]
        per = (a[i] - a[0]) / a[0]
        print(per * 100)

def active_learn(df1, first_item_index_of_each_category):
    train_idx = first_item_index_of_each_category

    data = df1.values[:,1:]
    target = df1['label'].values

    X_full = df1.values[:, 1:]
    y_full = df1['label'].values

    X_train = df1.values[:,1:][train_idx] #item from second column as the first column is the label..
    y_train = df1['label'].values[train_idx]


    X_pool = np.delete(data, train_idx, axis=0)
    y_pool = np.delete(target, train_idx)

    for i in range (1001, 1500):
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=X_train[:i], y_training=y_train[:i]
        )
        print('Initial prediction accuracy: %f' % learner.score(X_full,y_full))
    print("================================")
    print("================================")
    print("================================")
    print("================================")
    print("================================")
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_train[:1001], y_training=y_train[:1001]
    )

    n_queries = 502
    performance_array = []
    for idx in range(n_queries):
        query_idx, query_instance = learner.query(X_pool)
        learner.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
        learner_score = learner.score(data, target)
        # print('Accuracy after query no. %d: %f' % (idx + 1, learner_score))
        print('%f' % (learner_score))
    # clf = RandomForestClassifier()
    # clf.fit(X_train, y_train)
    # print(clf.score(X_train, y_train))
    # print('%f' % learner.score(X_full, y_full))
    # index = 0
    # performance_array = []
    # while learner.score(X_full, y_full) < 0.90:
    #     stream_idx = np.random.choice(range(len(X_full)))
    #     if classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1)) >= 0.4:
    #         learner.teach(X_full[stream_idx].reshape(1, -1), y_full[stream_idx].reshape(-1, ))
    #         learner_score = learner.score(X_full, y_full)
    #         # print('Item no. %d queried, new accuracy: %f' % (stream_idx, learner_score))
    #         print('%f' % (learner_score))
    #         if index == 505:
    #             break
    #         if (index % 100 == 0):
    #             performance_array.append(learner_score)
    #         index = index + 1
    # percentage_increase(performance_array)


if __name__ == "__main__":
    df_for_normalize = pd.read_csv('../../data/training.csv')
    df_normalized, first_item_index_of_each_category = normalize_datagram(df_for_normalize, 1000) #1k of each cat
    df1_back = df_normalized.copy(deep=True)
    sorted_entropy_list = compute_fisher_sorted(df1_back)
    sorted_entropy_list.insert(0, 'label')
    dataset = df1_back
    dataset.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat'], axis=1, inplace=True)
    feature_list = [20] #[2,3,4,5, 6] # [5, 10, 20] # [2, 3, 5, 10, 20]
    for i in range(0, len(feature_list)):
        df1 = dataset[sorted_entropy_list[:feature_list[i]]]
        active_learn(df1, first_item_index_of_each_category)

plt.show()