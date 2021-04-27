import matplotlib
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

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

warnings.filterwarnings('always')

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
    pod = pd.DataFrame(fisher_score.items(),  columns=['manufacturer', 'cty'])
    sorted_features = list(fisher_score.keys())

    for f in fisher_score:
        print("%.5f" % fisher_score[f])
    print(json.dumps(fisher_score, indent=2))
    return pod


def compute_fisher_sorted_multi(df):
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
    # print(json.dumps(fisher_score, indent=2))
    for f in fisher_score:
        # print(f)
        print("%.5f" % fisher_score[f])
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
    # df = compute_fisher_sorted(df_normalized)
    df = compute_fisher_sorted_multi(df_normalized)
    # Draw plot
    fig, ax = plt.subplots(figsize=(20, 12), dpi=80)
    ax.tick_params(labelrotation=90)

    matplotlib.pyplot.sca(ax)
    plt.xticks(rotation=90)

    ax.vlines(x=df.index, ymin=0, ymax=df.cty, color='firebrick', alpha=0.7, linewidth=2)
    ax.scatter(x=df.index, y=df.cty, s=75, color='firebrick', alpha=0.7)

    # Title, Label, Ticks and Ylim
    ax.set_title('Lollipop Chart for Highway Mileage', fontdict={'size': 22})
    ax.set_ylabel('Miles Per Gallon')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.manufacturer.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size': 12})
    ax.set_ylim(0, 3.5)

    # Annotate
    for row in df.itertuples():
        ax.text(row.Index, row.cty + .5, s=round(row.cty, 5),
                fontsize=8)

    plt.show()
