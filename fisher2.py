import pandas as pd
import numpy as np

from enum import Enum

class IMPURITY_TYPE(Enum):
    GINI_INDEX = 0
    ENTROYPY = 1

def compute_impurity(feature, impurity_criterion):

    probs = feature.value_counts(normalize=True)

    if impurity_criterion == IMPURITY_TYPE.ENTROYPY:
        impurity = -1 * np.sum(np.log2(probs) * probs)
    elif impurity_criterion == IMPURITY_TYPE.GINI_INDEX:
        impurity = 1 - np.sum(np.square(probs))
    else:
        raise ValueError('Unknown impurity criterion')

    return (round(impurity, 3))


df = pd.read_csv('data/training.csv')

# let's do two quick examples.
# print('impurity using entropy:', compute_impurity(df, IMPURITY_TYPE.ENTROYPY))
# print('impurity using gini index:', compute_impurity(df, IMPURITY_TYPE.GINI_INDEX))
#
# for feature in df.drop(columns = 'id').columns:
#     #feature_info_gain = comp_feature_information_gain(df, 'vegetation', feature, split_criterion)
#     print('Feature name: ' + feature)
#     print('\t gini index: ' + str(compute_impurity(df[feature], IMPURITY_TYPE.GINI_INDEX)))
#     print('\t entropy: ' + str(compute_impurity(df[feature], IMPURITY_TYPE.ENTROYPY)))


print(df.head())

print(df.isnull().sum())