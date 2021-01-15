test_dict = { "geeks" : 7, "for" : 1, "geeks" : 2 }
for x in test_dict.keys():
    print(x)


from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

k_range = list(range(1,100))
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors = k_range, weights = weight_options)


knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(X,y)

print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


# 0.9800000000000001
# {'n_neighbors': 13, 'weights': 'uniform'}
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=None, n_neighbors=13, p=2,
#                     weights='uniform')