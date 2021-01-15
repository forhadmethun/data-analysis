```python


# how to test for an incorrect compute_impurity_criterion value:
# print('impurity using gini index:', compute_impurity(df['stream'], 'foo'))

def compute():
    df = pd.read_csv('data/training.csv')
    print(df.describe())


if __name__ == '__main__':
    compute()


```