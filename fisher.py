import scipy.io
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split
import numpy as np


mat = scipy.io.loadmat("COIL20.mat")

x = 0

X=mat['X']
print(X)

y = mat['Y'][:, 0]
print(y)


n_samples, n_features = np.shape(X)

print (n_samples, n_features)


n_samples, n_features = np.shape(X)

print (n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

score = fisher_score.fisher_score(X_train, y_train)

print(score)




