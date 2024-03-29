import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings('always')

models = [
    ('LR', LogisticRegression(max_iter=4000)),
    ('NB', GaussianNB()),
    ('SVM', SVC()),
    ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
]


def applyModels(X, y, n):
    print("=========== : " + str(n) + " : ===============")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    score_list = {
        'LR': [],
        'NB': [],
        'SVM': [],
        'DT': [],
        'KNN': []
    }
    for name, model in models:
        clf = model
        clf.fit(X_train, y_train)
        # accuracy = clf.score(X_test, y_test)
        score = cross_val_score(clf, X, y, cv=5)
        mean_score = score.mean()
        print(name, mean_score)
        preds = clf.predict(X_test)
        # print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))
        # feature_list[name].append(n)
        score_list[name].append(mean_score)
    return score_list


"""
F = Ej,k(Pj(uj-u)^2)/Ej,k(Pj * oj^2)
uj = mean of data points belonging to class j 
oj = standard deviation of data points belonging to class j 
pj = fractions of data points belonging to class j
u = global mean of data on the feature being evaluated 
"""


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
    print(json.dumps(fisher_score, indent=2))
    sorted_features = list(fisher_score.keys())
    return sorted_features


def normalize_datagram(df, n):
    df.sample(frac=1)  # shuffle data

    # df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', 'label'])

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

    # list of index of the data more than n that will be removed
    removal_list = []
    for index, val in enumerate(header_feature):
        if (category_item_list[header_attack_cat[index]]['count'] > n):
            # category_item_list[header_attack_cat[index]]['index'].append(index)
            removal_list.append(index)
        category_item_list[header_attack_cat[index]]['count'] = category_item_list[header_attack_cat[index]][
                                                                    'count'] + 1

    i = 0
    # for category in attack_categories:
    #     df = df.drop(df.index[category_item_list[header_attack_cat[index]]['index']])
    df.drop(df.index[removal_list], inplace=True)
    return df


if __name__ == "__main__":
    # df = pd.read_csv('../../data/training.csv')
    # df_back = pd.read_csv('../../data/training.csv')
    df_for_normalize = pd.read_csv('../../data/training.csv')
    df_normalized = normalize_datagram(df_for_normalize, 1000)
    df_back = df_normalized.copy(deep=True)
    sorted_features = compute_fisher_sorted(df_normalized)
    feature_list = []
    score_list = []
    dt_score_list = []
    svm_score_list = []
    lr_score_list = []
    fisher_list = [3, 10, 25]
    fl = {
        'LR': [],
        'NB': [],
        'SVM': [],
        'DT': [],
        'KNN': []
    }


    header_attack_cat = df_for_normalize['attack_cat'].tolist()
    attack_categories = set(header_attack_cat)

    dictionary = {

    }
    df_back_current = df_normalized.copy(deep=True)
    # for m in range(0, len(attack_categories)):
    #     dictionary[m] = {}
    #     for n in range(m + 1, len(attack_categories)):
    #
            # dictionary[m][n] =


    for i in range(0, len(fisher_list)):
        df1 = df_normalized[sorted_features[:fisher_list[i]]]
        scaler = StandardScaler()
        scaler.fit(df1)
        sc_transform = scaler.transform(df1)
        sc_df = pd.DataFrame(sc_transform)
        X = sc_transform
        y = df_back['attack_cat']
        score = applyModels(X, y, fisher_list[i])
        for s in score:
            fl[s].append(score[s])
    x = np.array(fisher_list)  # X-axis points
    for s in fl:
        plt.plot(x, fl[s], label=s)
    plt.legend()
    plt.show()
