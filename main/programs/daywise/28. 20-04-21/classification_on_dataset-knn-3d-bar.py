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

models = [
    ('LR', LogisticRegression()),
    ('NB', GaussianNB()),
    ('SVM', SVC()),
    ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
]


def applyKnn(X_train, X_test, y_train, y_test, X, y, i, k = 3):
    print("=== KNN === : " + str(k))
    # k = 3
    # if i < k:
    #     k = i
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    score = knn.score(X, y)
    # print(confusion_matrix(y_test, preds))
    # print(classification_report(y_test, preds))
    # print(score)

    # clf = svm.SVC(kernel='linear', C=1, random_state=42)
# >> > scores = cross_val_score(clf, X, y, cv=5)
    score = cross_val_score(knn, X, y, cv=5)
    return i, score.mean()

def applyLogisticRegression(X_train, X_test, y_train, y_test , X, y, i):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    # score = logisticRegr.score(X, y)
    score = cross_val_score(logisticRegr, X, y, cv=5)
    score = score.mean()
    preds = logisticRegr.predict(X_test)
    # print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print(score)
    return i, score

#
# def applNNX_train, X_test, y_train, y_test , X, y, i):
#     logisticRegr = MLPClassifier()
#     logisticRegr.fit(X_train, y_train)
#     # score = logisticRegr.score(X, y)
#     score = cross_val_score(logisticRegr, X, y, cv=5)
#     score = score.mean()
#     preds = logisticRegr.predict(X_test)
#     # print(confusion_matrix(y_test, preds))
#     print(classification_report(y_test, preds))
#     print(score)
#     return i, score

def applyDT(X_train, X_test, y_train, y_test , X, y, i):
    print("=== DT=== : " + str(i))
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    # score = dt.score(X, y)

    score = cross_val_score(dt, X, y, cv=5)
    score = score.mean()

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
    # score = clf.score(X, y)
    score = cross_val_score(clf, X, y, cv=5)
    score = score.mean()
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

    # df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', 'label'])
    # sorted_features = list({
    #     "ct_dst_sport_ltm": 4.0895267043410675,
    #     "ct_dst_src_ltm": 2.2636905563106233,
    #     "ct_srv_dst": 2.242169670295672,
    #     "ct_src_dport_ltm": 2.2272399926781485,
    #     "ct_srv_src": 2.1315310499272138,
    #     "ct_dst_ltm": 1.7301922246444854,
    #     "sttl": 1.239129596789925,
    #     "ct_src_ltm": 1.105623796476083,
    #     "ct_state_ttl": 0.6190084974156483,
    #     "swin": 0.4538722829890125,
    #     "dwin": 0.43553083267016657,
    #     "dttl": 0.370427690585177,
    #     "rate": 0.277356097124606,
    #     "stcpb": 0.2331074609450747,
    #     "dtcpb": 0.2328042546481177,
    #     "dmean": 0.19734371359890013,
    #     "tcprtt": 0.19245502824432242,
    #     "dload": 0.18355796899257432,
    #     "ackdat": 0.18181511710719173,
    #     "synack": 0.15974392999977668,
    #     "smean": 0.08416136727375775,
    #     "ct_flw_http_mthd": 0.0754442329992394,
    #     "sload": 0.048261934367721394,
    #     "is_sm_ips_ports": 0.03531084706447892,
    #     "sinpkt": 0.03227088043802376,
    #     "dur": 0.022972694660332964,
    #     "dpkts": 0.019270709100707644,
    #     "is_ftp_login": 0.017161888151237554,
    #     "ct_ftp_cmd": 0.017161888151237554,
    #     "sjit": 0.01579916801110011,
    #     "trans_depth": 0.01328751915839373,
    #     "dloss": 0.012359773359293509,
    #     "djit": 0.011476584432656275,
    #     "spkts": 0.008408688518369968,
    #     "dbytes": 0.008239605705705054,
    #     "dinpkt": 0.007212881173681986,
    #     "sloss": 0.003993129033343524,
    #     "sbytes": 0.0035041218641487628,
    #     "response_body_len": 0.0011610434758266629
    # }.keys())
    return sorted_features

def normalize_datagram(df, n):

    df.sample(frac=1) # shuffle data

    # df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', 'label'])

    header_attack_cat = df['attack_cat'].tolist()
    attack_categories = set(header_attack_cat) # set of categories

    category_item_list = {}
    for category in attack_categories:
        category_item_list[category] = {
            "count" : 0,
            "index" : []
        }


    feature = 'attack_cat'
    header_feature = df[feature].tolist()

    #list of index of the data more than n that will be removed
    removal_list = []
    for index, val in enumerate(header_feature):
        if( category_item_list[header_attack_cat[index]]['count'] > n) :
            # category_item_list[header_attack_cat[index]]['index'].append(index)
            removal_list.append(index)
        category_item_list[header_attack_cat[index]]['count'] = category_item_list[header_attack_cat[index]]['count']  + 1

    i = 0
    # for category in attack_categories:
    #     df = df.drop(df.index[category_item_list[header_attack_cat[index]]['index']])
    df.drop(df.index[removal_list], inplace = True)
    return df
fig = plt.figure()
# ax = plt.axes(projection='3d')
ax = fig.add_subplot(111, projection='3d')

if __name__ == "__main__":

    df = pd.read_csv('../../data/training.csv')
    df_back = pd.read_csv('../../data/training.csv')
    df_for_normalize = pd.read_csv('../../data/training.csv')

    df_normalized = normalize_datagram(df_for_normalize, 1000)
    df_back = df_normalized.copy(deep = True)
    sorted_features = compute_fisher_sorted(df_normalized)
    # sorted_features = compute_fisher_sorted(df)

    i = 1
    j = 2

    feature_list = []
    knn_score_list = []
    dt_score_list = []
    svm_score_list = []
    lr_score_list = []

    fisher_list = [3, 6, 10, 15, 20, 25]
    x_list = []
    y_list = []
    z_list = []

    for i in range(0, len(fisher_list)):
        # df1 = df[sorted_features[:i]]
        df1 = df_normalized[sorted_features[:fisher_list[i]]]
        x = 0
        scaler = StandardScaler()
        scaler.fit(df1)
        sc_transform = scaler.transform(df1)
        sc_df = pd.DataFrame(sc_transform)
        X = sc_transform
        y = df_back['attack_cat']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        klist = [3, 5, 7, 9, 11]
        for kl in klist:
            i, score = applyKnn(X_train, X_test, y_train, y_test, X, y, i, kl)
            feature_list.append(fisher_list[i])
            knn_score_list.append(score)
            x_list.append(fisher_list[i])
            y_list.append(kl)
            z_list.append(score)

        # i,score = applyLogisticRegression(X_train, X_test, y_train, y_test , X, y, i )
        # lr_score_list.append(score)

        # i, score = applyDT(X_train, X_test, y_train, y_test, X, y, i)
        # dt_score_list.append(score)

    # define data values
    # x = np.array(feature_list)  # X-axis points
    # y_knn = np.array(knn_score_list)  # Y-axis points
    # y_dt = np.array(dt_score_list)
    # # y_svm = np.array(svm_score_list)
    # # y_lr = np.array(lr_score_list)
    #
    # plt.plot(x, y_knn, label="knn")
    # plt.plot(x, y_dt, label="dt")
    # # plt.plot(x, y_svm, label = "svm")
    # # plt.plot(x, y_lr, label="lr")
    # plt.legend()

    # ax.plot3D(x_list, y_list, z_list, 'red')
    # ax.scatter(x_list, y_list, z_list, 'red')
    dx = np.ones(len(x_list))
    dy = np.ones(len(y_list))
    dz = [] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(0, len(z_list)):
        dz.append(i)
    width = depth = 1
    ax.bar3d(x_list, y_list, z_list, dx, dy, dz, shade=True)
    ax.set_xlabel('# of features')
    ax.set_ylabel('# of n of knn')
    ax.set_zlabel('score')
    # ax.view_init(60, 50)
    plt.show()
