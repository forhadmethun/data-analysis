import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('../data/training.csv')
df_back = pd.read_csv('../data/training.csv')

fisher_score = {}

for feature in df.drop(columns = ['id', 'proto', 'service', 'state', 'attack_cat', 'label']).columns:
    header_attack_cat = df['attack_cat'].tolist()
    attack_categories = set(df['attack_cat'].tolist())
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

i = 5
j = 7
for i in range(i, j +1):
    df1 = df[sorted_features[:i+1]]
    x = 0
    scaler = StandardScaler()
    scaler.fit(df1)
    sc_transform = scaler.transform(df1)
    sc_df = pd.DataFrame(sc_transform)
    X = sc_transform
    y = df_back['attack_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    logisticRegr = LogisticRegression()

    k_range = list(range(1, 7))
    weight_options = ["uniform", "distance"]

    param_grid = dict(n_neighbors=k_range, weights=weight_options)

    knn = KNeighborsClassifier()

    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)

    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)

