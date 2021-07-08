import pandas as pd
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
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
    ('RF', RandomForestClassifier()),
]

def applyModels(X, y, n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    score_list = {
        'LR': 0,
        'NB': 0,
        'SVM': 0,
        'DT': 0,
        'KNN': 0,
        'RF': 0,
    }
    for name, model in models:
        clf = model
        clf.fit(X_train, y_train)
        # accuracy = clf.score(X_test, y_test)
        score = cross_val_score(clf, X, y, cv=5)
        mean_score = score.mean()
        preds = clf.predict(X_test)
        # print(confusion_matrix(y_test, preds))
        print("===========================================================================================")
        print("=======================   ALGORITHM: " + name + "  ==================================================")
        print("=======================  FEATURE NO: " + str(n) + "  ===================================================")
        print("=================== CROSS VAL SCORE: "+ str(mean_score) +"  =================================")
        print("===========================================================================================")
        print("============================ CLASSIFICATION REPORT ========================================")
        print()
        # print(mean_score)
        print(classification_report(y_test, preds))
        # feature_list[name].append(n)


        # print(precision_score(y_test, preds, average="macro"))
        # mean_score = precision_score(y_test, preds, average="macro");

        # print(recall_score(y_test, preds, average="macro"))
        # mean_score = recall_score(y_test, preds, average="macro");
        #
        #
        # print(f1_score(y_test, preds, average="macro"))
        # mean_score = f1_score(y_test, preds, average="macro");


        score_list[name] = mean_score
        print()
        print()
        print("###########################################################################################")
        print()
        print()
    return score_list

def compute_fisher_sorted(df):
    print("====================================================================")
    print("======================= FISHER SCORE ===============================")
    print("====================================================================")
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
    return df


if __name__ == "__main__":
    df_for_normalize = pd.read_csv('../../data/training.csv')
    df_normalized = normalize_datagram(df_for_normalize, 5000)
    df_back = df_normalized.copy(deep=True)
    sorted_features = compute_fisher_sorted(df_normalized)
    feature_list = []
    score_list = []
    dt_score_list = []
    svm_score_list = []
    lr_score_list = []
    fisher_list = [3, 6, 9, 15, 20, 25]
    fl = {
        'LR': [],
        'NB': [],
        'SVM': [],
        'DT': [],
        'KNN': [],
        'RF': []
    }
    header_attack_cat = df_for_normalize['attack_cat'].tolist()
    attack_categories = set(header_attack_cat)
    for i in range(0, len(fisher_list)):
        df1 = df_normalized[sorted_features[:fisher_list[i]]]
        scaler = StandardScaler()
        scaler.fit(df1)
        sc_transform = scaler.transform(df1)
        sc_df = pd.DataFrame(sc_transform)
        X = sc_transform
        y = df_back['label']
        score = applyModels(X, y, fisher_list[i])
        for s in score:
            fl[s].append(score[s])
    x = np.array(fisher_list)  # X-axis points
    bla  = 0
    for s in fl:
        length = len(fisher_list)
        # plt.bar([p+bla for p in x], fl[s],width=1 / (length + 1), label=s)
        plt.bar([p+bla for p in x], fl[s],width= .4, label=s)
        # bla = bla + 1 / (length + 1)
        bla = bla + .25
    plt.legend()
    plt.show()
