import pandas as pd
from numpy import *
import json

df = pd.read_csv('data/training.csv')

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

