import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def applyLogisticRegression(X_train, X_test, y_train, y_test , X, y, i):
    k = 3
    if i < k:
        k = i
    logisticRegr = LogisticRegression(max_iter = k)
    logisticRegr.fit(X_train, y_train)
    score = logisticRegr.score(X, y)
    preds = logisticRegr.predict(X_test)
    # print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print(score)
    return i, score

def applyKnn(X_train, X_test, y_train, y_test , X, y, i):
    print("=== KNN === : " + str(i))
    k = 3# //todo: need to find the value of k
    if i < k:
        k = i
    # k_range = list(range(1, 7))
    # weight_options = ["uniform", "distance"]
    #
    # param_grid = dict(n_neighbors=k_range, weights=weight_options)
    #
    # knn = KNeighborsClassifier()
    # grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    # grid.fit(X, y)
    #
    # print(grid.best_score_)
    # print(grid.best_params_)
    # print(grid.best_estimator_)


    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    score = knn.score(X, y)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print(score)
    return i, score

def applyDT(X_train, X_test, y_train, y_test , X, y, i):
    print("=== DT=== : " + str(i))
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    score = dt.score(X, y)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print(score)
    return i, score

def applySVM(X_train, X_test, y_train, y_test , X, y, i):
    if i <= 1 :
        return 1, []
    print("=== SVM === :" + str(i))
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    score = clf.score(X, y)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print(score)
    return i, score

"""
F = Ej,k(Pj(uj-u)^2)/Ej,k(Pj * oj^2)
uj = mean of data points belonging to class j 
oj = standard deviation of data points belonging to class j 
pj = fractions of data points belonging to class j
u = global mean of data on the feature being evaluated 
"""

if __name__ == "__main__":
    df = pd.read_csv('../../data/training.csv')
    df_back = pd.read_csv('../../data/training.csv')
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
    i = 1
    j = 30
    feature_list = []
    score_list = []
    dt_score_list = []
    svm_score_list = []
    lr_score_list = []
    for i in range(i, j + 1):
        df1 = df[sorted_features[:i + 1]]
        x = 0
        scaler = StandardScaler()
        scaler.fit(df1)
        sc_transform = scaler.transform(df1)
        sc_df = pd.DataFrame(sc_transform)
        X = sc_transform
        y = df_back['attack_cat']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        i,score = applyLogisticRegression(X_train, X_test, y_train, y_test , X, y, i )
        lr_score_list.append(score)

        i, score = applyKnn(X_train, X_test, y_train, y_test , X, y, i)
        feature_list.append(i)
        score_list.append(score)

        i, score = applyDT(X_train, X_test, y_train, y_test, X, y, i)
        dt_score_list.append(score)

        # i, score = applySVM(X_train, X_test, y_train, y_test, X, y, i)
        # svm_score_list.append(score)

        # logisticRegr = LogisticRegression()
        # logisticRegr.fit(X_train, y_train)
        # score = logisticRegr.score(X, y)
        # preds = logisticRegr.predict(X_test)
        # print(confusion_matrix(y_test, preds))
        # print(classification_report(y_test, preds))
        # print(score)

    # define data values
    x = np.array(feature_list)  # X-axis points
    y_knn = np.array(score_list)  # Y-axis points
    y_dt = np.array(dt_score_list)
    # y_svm = np.array(svm_score_list)
    y_lr = np.array(lr_score_list)

    plt.plot(x, y_knn, label = "knn")
    plt.plot(x, y_dt, label = "dt")
    # plt.plot(x, y_svm, label = "svm")
    plt.plot(x, y_lr, label = "lr")
    plt.legend()
    plt.show()