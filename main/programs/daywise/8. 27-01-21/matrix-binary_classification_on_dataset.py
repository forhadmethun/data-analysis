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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import warnings

warnings.filterwarnings('always')

models = [
    # ('LR', LogisticRegression(max_iter=4000)),
    # ('NB', GaussianNB()),
    # ('SVM', SVC()),
    # ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
]

def applyModels(X, y, n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    score_list = {
        # 'LR': 0,
        # 'NB': 0,
        # 'SVM': 0,
        'DT': 0,
        # 'KNN': 0
    }
    for name, model in models:
        clf = model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score =  recall_score(y_test, y_pred, average="macro")
        score_list[name] = score #mean_score
    return score_list

def compute_fisher_sorted(df):
    # print("====================================================================")
    # print("======================= FISHER SCORE ===============================")
    # print("====================================================================")
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
    # print(json.dumps(fisher_score, indent=2))
    sorted_features = list(fisher_score.keys())
    return sorted_features


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



def render_datagram_for_two_class(df, attack_categories,  m, n):
    df.sample(frac=1)
    # header_attack_cat = df['attack_cat'].tolist()
    # attack_categories = list(set(header_attack_cat)) # set of categories
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
        if header_feature[index] == m or header_feature[index] == n:
            continue
        removal_list.append(index)
    df.drop(df.index[removal_list], inplace=True)
    return df


if __name__ == "__main__":
    df = pd.read_csv('../../data/training.csv')
    df_for_normalize = pd.read_csv('../../data/training.csv')

    df_normalized = normalize_datagram(df_for_normalize, 1000)

    df_back = df_normalized.copy(deep=True)

    sorted_features = compute_fisher_sorted(df_normalized)
    feature_list = []
    score_list = []
    dt_score_list = []
    svm_score_list = []
    lr_score_list = []
    fisher_list = [10] #Todo, try for all fisher.. first with one!! 10]#3,25

    header_attack_cat = df_for_normalize['attack_cat'].tolist()
    attack_categories = list(set(header_attack_cat))

    cols_count = len(attack_categories)
    rows_count = cols_count
    result_matrix = [[0 for x in range(cols_count)] for x in range(rows_count)]

    df1_back = df_back.copy(deep=True)
    for m in range(0, len(attack_categories)):
        for n in range(m + 1, len(attack_categories)):
            fl = {
                # 'LR': [],
                # 'NB': [],
                # 'SVM': [],
                'DT': [],
                # 'KNN': []
            }
            df1_current = df1_back.copy(deep=True)
            df1_current = render_datagram_for_two_class(df1_current, attack_categories, attack_categories[m], attack_categories[n])
            for i in range(0, len(fisher_list)):
                df1_current_back = df1_current.copy(deep=True)
                df1 = df1_current[sorted_features[:fisher_list[i]]]
                scaler = StandardScaler()
                scaler.fit(df1)
                sc_transform = scaler.transform(df1)
                sc_df = pd.DataFrame(sc_transform)
                X = sc_transform
                y = df1_current_back['attack_cat']
                score = applyModels(X, y, fisher_list[i])
                for s in score:
                    fl[s].append(score[s])

                result_matrix[m][n] = score['DT']
                # print("--------------------------------------------------")
                # print(attack_categories[m] + " VS " + attack_categories[n])
                # print("SAMPLE SIZE: " + str(len(df1)))
                # print("CROSS VALIDATION SCORE: ")
                # print(score)
                # print("--------------------------------------------------")
                # print()
                # print()

    #         x = np.array(fisher_list)  # X-axis points
    #         bla  = 0
    #         for s in fl:
    #             plt.bar([p+bla for p in x], fl[s], label=s)
    #             bla = bla + 0.25
    # plt.legend()
    # plt.show()

# for i in range(0, len(attack_categories)):
#     mat_elem = {}
#     for j in range(i+1, len(attack_categories)):
#         result_matrix[i][j] = str(i) + " " + str(j)
        # mat[attack_categories]
print('%15s' % '', end= " ")
for i in range(0, len(attack_categories)):
    print('%15s' % attack_categories[i], end= " ")
print()

for i in range(0, len(attack_categories)):
    print('%15s' % attack_categories[i], end=" ")
    for j in range(0, len(attack_categories)):
        if( result_matrix[i][j] == 0):
            print('%15s' % "-", end = " ")
        else:
            print('%15s' % (str(result_matrix[i][j])[:5]), end = " ")
    print()
print(attack_categories)


numpy_reslt_mat = np.array(result_matrix)
plt.imshow(numpy_reslt_mat, cmap='hot', interpolation='nearest')
plt.show()