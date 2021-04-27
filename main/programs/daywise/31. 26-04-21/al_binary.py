import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modAL import Committee
from modAL.models import ActiveLearner
from modAL.uncertainty import margin_sampling, classifier_uncertainty, classifier_margin
from numpy import *
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('always')

# extracting 1k data of each category from the dataset
category_list = ["Fuzzers", "Exploits", "Worms", "Shellcode", "Generic", "Analysis", "Backdoor", "DoS",
                 "Reconnaissance",
                 "Normal"]

def normalize_datagram(df, n):
    df.sample(frac=1)
    header_attack_cat = df['label'].tolist()
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
    feature = 'label'
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
    header_attack_cat = df['label'].tolist()
    for index, val in enumerate(header_feature):
        normalized_category_item_list[header_attack_cat[index]]['count'] = \
            normalized_category_item_list[header_attack_cat[index]]['count'] + 1
        normalized_category_item_list[header_attack_cat[index]]['index'].append(index)

    first_item_index_of_each_category = []
    for key, value in normalized_category_item_list.items():
        for i in range(0, 50):
            if (i < len(value['index'])):
             first_item_index_of_each_category.append(value['index'][i])

    # for index, val in enumerate(header_feature):
    # for i, l in enumerate(category_list):
    #     df.loc[df['attack_cat'] == l, ['label']] = i
    return df, first_item_index_of_each_category

def compute_fisher_sorted(df):
    fisher_score = {}
    for feature in df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', 'label']).columns:
        header_attack_cat = df['label'].tolist()
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

    # simple_rf(data, target, X_train, y_train, X_full, y_full, train_idx)
    al_pool(data, target, X_train, y_train, X_full, y_full, train_idx)
    # al_stream(data, target, X_train, y_train, X_full, y_full, train_idx)
    # al_qbc(data, target, X_train, y_train, X_full, y_full, train_idx)
    # uncertainty_values(data, target, X_train, y_train, X_full, y_full, train_idx)

def simple_rf(data, target, X_train, y_train, X_full, y_full, train_idx):
    # print("START: RF")
    for i in range (201, 1701):
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=X_train[:i], y_training=y_train[:i]
        )
        print(' %0.3f' % learner.score(X_full,y_full), end=",")
    # print("END: RF")

def al_pool(data, target, X_train, y_train, X_full, y_full, train_idx):
    X_pool = np.delete(data, train_idx, axis=0)
    y_pool = np.delete(target, train_idx)
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_train, y_training=y_train
    )

    n_queries = 220
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
        # print('Accuracy after query no. %d: %f' % (idx + 1, learner_wscore))
        print('%0.3f' % (learner_score), end=",")

def al_stream(data, target, X_train, y_train, X_full, y_full, train_idx):
    # initializing the active learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy= margin_sampling,
        X_training=X_train, y_training=y_train
    )

    # print('Initial prediction accuracy: %f' % learner.score(X_full, y_full))
    index = 0
    # learning until the accuracy reaches a given threshold
    while learner.score(X_full, y_full) < 0.90:
        stream_idx = np.random.choice(range(len(X_full)))
        if classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1)) >= 0.4:
            learner.teach(X_full[stream_idx].reshape(1, -1), y_full[stream_idx].reshape(-1, ))
            learner_score = learner.score(X_full, y_full)
            # print('Item no. %d queried, new accuracy: %f' % (stream_idx, learner_score))
            print('%0.3f' % (learner_score) , end=",")
            if index == 300:
                break
            index = index + 1

def al_qbc(data, target, X_train, y_train, X_full, y_full, train_idx):
    # print("START: Q")
    X_pool = deepcopy(X_full)
    y_pool = deepcopy(y_full)

    # initializing Committee members
    n_members = 2
    learner_list = list()

    for member_idx in range(n_members):
        # initial training data
        # n_initial = 5
        # train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
        # X_train = X_pool[train_idx]
        # y_train = y_pool[train_idx]

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

    # print('Committee initial predictions, accuracy = %1.3f' % committee.score(data, target))
    # print('%1.3f' % committee.score(data, target))

    n_queries = 500
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
        # print('Committee %d th query predictions, accuracy = %1.3f' % (idx , learner_score))
        print('%0.3f' % (learner_score), end=",")
    # print("END: Q")

def uncertainty_values(data, target, X_train, y_train, X_full, y_full, train_idx):
    print("START: ST")
    # initializing the active learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=margin_sampling,
        X_training=X_train, y_training=y_train
    )
    print('%f' % learner.score(X_full, y_full))
    index = 0
    # learning until the accuracy reaches a given threshold
    while learner.score(X_full, y_full) < 0.90:
        stream_idx = np.random.choice(range(len(X_full)))
        if classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1)) >= 0.4:

            print("[ %1.3f, %1.3f]" %   (classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1))[0], classifier_margin(learner, X_full[stream_idx].reshape(1, -1))[0] ))

            learner.teach(X_full[stream_idx].reshape(1, -1), y_full[stream_idx].reshape(-1, ))
            learner_score = learner.score(X_full, y_full)
            # print('Item no. %d queried, new accuracy: %f' % (stream_idx, learner_score))
            # print('%f' % (learner_score))
            if index == 50:
                break
            index = index + 1
    print("START: ST")

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