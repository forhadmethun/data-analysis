import collections
import math
from collections import Counter
from scipy import stats

def entropy1():
    s=range(0,256)

    # calculate probability for each byte as number of occurrences / array length
    probabilities = [n_x/len(s) for x,n_x in collections.Counter(s).items()]
    # [0.00390625, 0.00390625, 0.00390625, ...]

    # calculate per-character entropy fractions
    e_x = [-p_x*math.log(p_x,2) for p_x in probabilities]
    # [0.03125, 0.03125, 0.03125, ...]

    # sum fractions to obtain Shannon entropy
    entropy = sum(e_x)

    print(entropy)

# entropy1()
def entropy2():
    labels = [0.9, 0.09, 0.1]
    x = stats.entropy(list(Counter(labels).keys()), base=2)
    print(x)

entropy2()