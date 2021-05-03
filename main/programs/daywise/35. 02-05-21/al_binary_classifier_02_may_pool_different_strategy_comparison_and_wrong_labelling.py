from copy import deepcopy
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modAL import Committee
from modAL.models import ActiveLearner
from modAL.uncertainty import margin_sampling, classifier_uncertainty, classifier_margin, entropy_sampling, \
    classifier_entropy, uncertainty_sampling
from numpy import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from modAL.disagreement import vote_entropy_sampling, consensus_entropy_sampling, max_disagreement_sampling, \
    max_std_sampling
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Method:
    rf = 'rf'
    pool = 'pool',
    stream = 'stream',
    qbc = 'qbc'
    uncertainty_pool = 'uncertainty_pool'
    margin_pool = 'margin_pool'
    entropy_pool = 'entropy_pool'


class BinaryAL:
    def __init__(self, initial_point, query_number):
        self.init(initial_point, query_number)

    def init(self, initial_point, query_number):
        self.initial_point = initial_point
        self.query_number = query_number
        self.df_for_normalize = pd.read_csv('../../data/training.csv')
        self.category_list = ["Fuzzers", "Exploits", "Worms", "Shellcode", "Generic", "Analysis", "Backdoor", "DoS",
                              "Reconnaissance",
                              "Normal"]
        self.df_normalized, self.first_item_index_of_each_category = self.normalize_datagram(self.df_for_normalize,
                                                                                             1000)  # 1k of each cat
        self.df1_back = self.df_normalized.copy(deep=True)
        self.sorted_fisher_list = self.compute_fisher_sorted(self.df1_back)
        self.sorted_fisher_list.insert(0, 'label')
        self.dataset = self.df1_back
        self.dataset.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat'], axis=1, inplace=True)
        self.feature_list = [20]  # [2,3,4,5, 6] # [5, 10, 20] # [2, 3, 5, 10, 20]
        self.df1 = self.dataset[self.sorted_fisher_list[:self.feature_list[0]]]

    def normalize_datagram(self, df, n):
        df.sample(frac=1)
        header_attack_cat = df['label'].tolist()
        # header_attack_cat = df['attack_cat'].tolist()
        attack_categories = set(header_attack_cat)  # set of categories
        category_item_list = {}
        normalized_category_item_list = {}
        for category in attack_categories:
            category_item_list[category] = {
                "count": 0,
                "index": []
            }
            normalized_category_item_list[category] = {
                "count": 0,
                "index": []
            }
        feature = 'label'
        # feature = 'attack_cat'
        header_feature = df[feature].tolist()
        removal_list = []
        for index, val in enumerate(header_feature):
            if (category_item_list[header_attack_cat[index]]['count'] > n):
                removal_list.append(index)
            category_item_list[header_attack_cat[index]]['count'] = \
                category_item_list[header_attack_cat[index]]['count'] + 1
            # category_item_list[header_attack_cat[index]]['index'].append(index)
        df.drop(df.index[removal_list], inplace=True)

        # normalized calculation after removing the extra data
        header_feature = df[feature].tolist()
        header_attack_cat = df['label'].tolist()
        # header_attack_cat = df['attack_cat'].tolist()
        for index, val in enumerate(header_feature):
            normalized_category_item_list[header_attack_cat[index]]['count'] = \
                normalized_category_item_list[header_attack_cat[index]]['count'] + 1
            normalized_category_item_list[header_attack_cat[index]]['index'].append(index)

        first_item_index_of_each_category = []
        for key, value in normalized_category_item_list.items():
            for i in range(0, self.initial_point):
                if (i < len(value['index'])):
                    first_item_index_of_each_category.append(value['index'][i])

        # for index, val in enumerate(header_feature):
        for i, l in enumerate(self.category_list):
            df.loc[df['label'] == l, ['label']] = i
            # df.loc[df['attack_cat'] == l, ['label']] = i
        return df, first_item_index_of_each_category

    def compute_fisher_sorted(self, df):
        fisher_score = {}
        for feature in df.drop(columns=['id', 'proto', 'service', 'state', 'attack_cat', 'label']).columns:
            header_attack_cat = df['label'].tolist()  # this is for binary class classifier
            # header_attack_cat = df['attack_cat'].tolist()  # this is for multi class classifier
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
        sorted_features = list(fisher_score.keys())
        return sorted_features

    def performance_measure(self, learner, X_full, y_full):
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.30)
        y_predict = learner.predict(X_test)
        # precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)

        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        fscore = f1_score(y_test, y_predict)
        support = 0

        return precision, recall, fscore, support, accuracy

    def active_learn(self, df1, first_item_index_of_each_category, method):
        train_idx = first_item_index_of_each_category

        data = df1.values[:, 1:]
        target = df1['label'].values

        X_full = df1.values[:, 1:]
        y_full = df1['label'].values

        X_train = df1.values[:, 1:][train_idx]  # item from second column as the first column is the label..
        y_train = df1['label'].values[train_idx]

        X_pool = np.delete(data, train_idx, axis=0)
        y_pool = np.delete(target, train_idx)

        if method == Method.pool:
            print("--- pool ---")
            return self.al_pool(data, target, X_train, y_train, X_full, y_full, train_idx)

        if method == Method.margin_pool:
            print("--- pool ---")
            return self.al_pool_margin(data, target, X_train, y_train, X_full, y_full, train_idx)

        if method == Method.entropy_pool:
            print("--- pool ---")
            return self.al_pool_entropy(data, target, X_train, y_train, X_full, y_full, train_idx)

        if method == Method.stream:
            print("--- stream ---")
            return self.al_stream(data, target, X_train, y_train, X_full, y_full, train_idx)
        if method == Method.qbc:
            print("--- qbc ---")
            return self.al_qbc(data, target, X_train, y_train, X_full, y_full, train_idx)
        if method == Method.rf:
            print("--- rf ---")
            return self.simple_rf(data, target, X_train, y_train, X_full, y_full, train_idx)
        # if method == Method.rf:
        #     self.uncertainty_values(data, target, X_train, y_train, X_full, y_full, train_idx)

    def simple_rf(self, data, target, X_train, y_train, X_full, y_full, train_idx):
        # print("START: RF")
        acc = []
        for i in range(self.initial_point * 2, self.initial_point * 2 + self.query_number):
            learner = ActiveLearner(
                estimator=RandomForestClassifier(),
                X_training=X_train[:i], y_training=y_train[:i]
            )
            score = learner.score(X_full, y_full)
            acc.append(score)
            print(' %0.3f' % score, end=",")
        # print("END: RF")
        return acc

    def al_pool(self, data, target, X_train, y_train, X_full, y_full, train_idx):
        return self.al_pool(data, target, X_train, y_train, X_full,
                            y_full, train_idx, RandomForestClassifier(), uncertainty_sampling)

    ## START: POOL
    def al_pool(self, data, target, X_train, y_train, X_full, y_full, train_idx, classifier, sampling_strategy):
        acc = []
        pre = []
        rec = []
        fs = []
        X_pool = np.delete(data, train_idx, axis=0)
        y_pool = np.delete(target, train_idx)
        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=sampling_strategy,
            X_training=X_train, y_training=y_train
        )

        n_queries = self.query_number
        # n_queries = 1500
        for idx in range(n_queries):
            query_idx, query_instance = learner.query(X_pool)
            learner.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )
            # remove queried instance from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)
            learner_score = learner.score(data, target)
            # print('Accuracy after query no. %d: %f' % (idx + 1, learner_wscore))
            precision, recall, fscore, support, accuracy = self.performance_measure(learner, X_full, y_full)
            # learner_score = fscore
            acc.append(accuracy)
            pre.append(precision)
            rec.append(recall)
            fs.append(fscore)
            print('%0.3f' % (learner_score), end=",")
        return fs
    def al_pool_proba(self, data, target, X_train, y_train, X_full, y_full, train_idx, classifier, sampling_strategy, proba):
        acc = []
        pre = []
        rec = []
        fs = []
        X_pool = np.delete(data, train_idx, axis=0)
        y_pool = np.delete(target, train_idx)
        learner = ActiveLearner(
            estimator=classifier,
            query_strategy=sampling_strategy,
            X_training=X_train, y_training=y_train
        )

        n_queries = self.query_number
        # n_queries = 1500
        for idx in range(n_queries):
            query_idx, query_instance = learner.query(X_pool)

            labeled_y = y_pool[query_idx].reshape(1, )
            rand_int = randint(0, 100)
            if(rand_int <= proba):
                if( y_pool[query_idx][0] == 1):
                    y_pool[query_idx][0] = 0
                    labeled_y = np.array((0)).reshape(1,)
                else:
                    y_pool[query_idx][0] = 1
                    labeled_y = np.array((1)).reshape(1, )

            learner.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=labeled_y
            )
            # remove queried instance from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)
            learner_score = learner.score(data, target)
            # print('Accuracy after query no. %d: %f' % (idx + 1, learner_wscore))
            precision, recall, fscore, support, accuracy = self.performance_measure(learner, X_full, y_full)
            # learner_score = fscore
            acc.append(accuracy)
            pre.append(precision)
            rec.append(recall)
            fs.append(fscore)
            print('%0.3f' % (learner_score), end=",")
        return acc, pre, rec, fs
    ## END: POOL

    def al_stream(self, data, target, X_train, y_train, X_full, y_full, train_idx):
        # initializing the active learner
        acc = []
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            query_strategy=margin_sampling,
            X_training=X_train, y_training=y_train
        )

        # print('Initial prediction accuracy: %f' % learner.score(X_full, y_full))
        index = 0
        # learning until the accuracy reaches a given threshold
        while learner.score(X_full, y_full) < 0.90:
            stream_idx = np.random.choice(range(len(X_full)))
            if classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1)) >= 0.2:
                learner.teach(X_full[stream_idx].reshape(1, -1), y_full[stream_idx].reshape(-1, ))
                learner_score = learner.score(X_full, y_full)
                # print('Item no. %d queried, new accuracy: %f' % (stream_idx, learner_score))
                print('%0.3f' % (learner_score), end=",")
                if index == self.query_number:
                    break
                index = index + 1
                acc.append(learner_score)
        return acc

    def al_qbc(self, data, target, X_train, y_train, X_full, y_full, train_idx):
        self.al_qbc(data, target, X_train, y_train, X_full, y_full, train_idx, consensus_entropy_sampling)

    ## START: QBC
    def al_qbc(self, data, target, X_train, y_train, X_full, y_full, train_idx, committee_strategy):
        acc = []
        pre = []
        rec = []
        fs = []

        X_pool = deepcopy(X_full)
        y_pool = deepcopy(y_full)

        # initializing Committee members
        n_members = 2
        learner_list = list()

        for member_idx in range(n_members):
            # initial training data
            # n_initial = 5
            # train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
            # X_train = X_pool[train_idx]
            # y_train = y_pool[train_idx]

            # creating a reduced copy of the data with the known instances removed
            X_pool = np.delete(X_pool, train_idx, axis=0)
            y_pool = np.delete(y_pool, train_idx)

            # initializing learner
            learner = ActiveLearner(
                estimator=RandomForestClassifier(),
                # query_strategy=vote_entropy_sampling,

                X_training=X_train, y_training=y_train
            )
            learner_list.append(learner)
            # assembling the committee
        committee = Committee(
            learner_list=learner_list,
            query_strategy=committee_strategy
        )

        # print('Committee initial predictions, accuracy = %1.3f' % committee.score(data, target))
        # print('%1.3f' % committee.score(data, target))

        n_queries = self.query_number
        for idx in range(n_queries):
            query_idx, query_instance = committee.query(X_pool)
            committee.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )
            # remove queried instance from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)
            # learner_score = committee.score(data, target)
            # print('Committee %d th query predictions, accuracy = %1.3f' % (idx , learner_score))
            precision, recall, fscore, support, accuracy = self.performance_measure(learner, X_full, y_full)
            learner_score = accuracy
            acc.append(learner_score)
            pre.append(precision)
            rec.append(recall)
            fs.append(fscore)
            print('%0.3f' % (learner_score), end=",")
        return fs
        # print("END: Q")
    ## END: QBC

    def uncertainty_values(self, data, target, X_train, y_train, X_full, y_full, train_idx):
        print("START: ST")
        # initializing the active learner
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            query_strategy=margin_sampling,
            X_training=X_train, y_training=y_train
        )
        print('%f' % learner.score(X_full, y_full))
        index = 0
        # learning until the accuracy reaches a given threshold
        while learner.score(X_full, y_full) < 0.90:
            stream_idx = np.random.choice(range(len(X_full)))
            if classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1)) >= 0.4:

                print("[ %1.3f, %1.3f]" % (classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1))[0],
                                           classifier_margin(learner, X_full[stream_idx].reshape(1, -1))[0]))

                learner.teach(X_full[stream_idx].reshape(1, -1), y_full[stream_idx].reshape(-1, ))
                learner_score = learner.score(X_full, y_full)
                # print('Item no. %d queried, new accuracy: %f' % (stream_idx, learner_score))
                # print('%f' % (learner_score))
                if index == 50:
                    break
                index = index + 1
        print("START: ST")

    def plotter(self, y1, y2, y3, y4, how_many_max_instances, start_index):
        x = []
        for i in range(0, how_many_max_instances):
            x.append(i)
        # y1 = [ ]
        plt.plot(x[start_index:how_many_max_instances], y1[start_index:how_many_max_instances],
                 label="Least confident uncertainty sampling")

        # y2 = [ ]
        plt.plot(x[start_index:how_many_max_instances], y2[start_index:how_many_max_instances],
                 label="Max margin uncertainty sampling")

        # y3 = [ ]
        plt.plot(x[start_index:how_many_max_instances], y3[start_index:how_many_max_instances],
                 label="Max entropy unceratinty sampling")

        # y4 = []
        plt.plot(x[start_index:how_many_max_instances], y4[start_index:how_many_max_instances],
                 label="Query by committee sampling")

        plt.xlabel('Query Instances')
        plt.ylabel('F1 Score')
        plt.title('Pool based Active learning accuracy performance on binary case')
        plt.legend()
        plt.show()

    def dumb_plotter_qbc(self, performance_measure, start_index):

        for key in performance_measure:
            y = performance_measure[key]
            x = []
            for i in range(0, len(y)):
                x.append(i)
            # y1 = [ ]
            plt.plot(x[start_index:len(y ) - 1], y[start_index:len(y) - 1],
                     label=key)

        plt.xlabel('Query Iteration')
        plt.ylabel('Performance measure(precision)')
        plt.title('Query By Committee(different strategy)performance measure on binary case')
        plt.legend()
        plt.show()

    def dumb_plotter_pool(self, performance_measure, start_index, ylabel):

        for key in performance_measure:
            y = performance_measure[key]
            x = []
            for i in range(0, len(y)):
                x.append(i)
            # y1 = [ ]
            plt.plot(x[start_index:len(y) - 1], y[start_index:len(y) - 1],
                     label=key)

        plt.xlabel('Query Iteration')
        plt.ylabel(ylabel)
        plt.title('Pool based selection strategy on binary case')
        plt.legend()
        plt.show()

    def dumb_plotter_proba(self, performance_measure, start_index, proba):

        for key in performance_measure:
            y = performance_measure[key]
            x = []
            for i in range(0, len(y)):
                x.append(i)
            # y1 = [ ]
            plt.plot(x[start_index:len(y) - 1], y[start_index:len(y) - 1],
                     label=key)

        plt.xlabel('Query Iteration')
        plt.ylabel('Performance')
        plt.title('Pool based selection strategy on binary case(' + str(proba) + '% wrong labeling' + ')')
        plt.legend()
        plt.show()


    def learnAndPlot(self):
        # for i in range(0, len(self.feature_list)):
        # df1 = self.dataset[self.sorted_entropy_list[:self.feature_list[i]]]

        y1 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.qbc)
        # y1 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.pool)
        self.init(self.initial_point, self.query_number)

        y2 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.margin_pool)
        self.init(self.initial_point, self.query_number)

        y3 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.entropy_pool)
        self.init(self.initial_point, self.query_number)

        y4 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.qbc)
        self.init(self.initial_point, self.query_number)

        self.plotter(y1, y2, y3, y4, self.query_number - 1, 15)

    def learn_pool_proba(self, classifier, sampling_strategy, proba):

        train_idx = self.first_item_index_of_each_category

        data = self.df1.values[:, 1:]
        target = self.df1['label'].values

        X_full = self.df1.values[:, 1:]
        y_full = self.df1['label'].values

        X_train = self.df1.values[:, 1:][train_idx]  # item from second column as the first column is the label..
        y_train = self.df1['label'].values[train_idx]
        return self.al_pool_proba(data, target, X_train, y_train, X_full, y_full, train_idx, classifier, sampling_strategy, proba)

    def learn_and_plot_pool(self, classifier, sampling_strategy):

        train_idx = self.first_item_index_of_each_category

        data = self.df1.values[:, 1:]
        target = self.df1['label'].values

        X_full = self.df1.values[:, 1:]
        y_full = self.df1['label'].values

        X_train = self.df1.values[:, 1:][train_idx]  # item from second column as the first column is the label..
        y_train = self.df1['label'].values[train_idx]
        return self.al_pool(data, target, X_train, y_train, X_full, y_full, train_idx, classifier, sampling_strategy)

    def learn_and_plot_qbc(self, committee_strategy):

        train_idx = self.first_item_index_of_each_category

        data = self.df1.values[:, 1:]
        target = self.df1['label'].values

        X_full = self.df1.values[:, 1:]
        y_full = self.df1['label'].values

        X_train = self.df1.values[:, 1:][train_idx]  # item from second column as the first column is the label..
        y_train = self.df1['label'].values[train_idx]
        return self.al_qbc(data, target, X_train, y_train, X_full, y_full, train_idx, committee_strategy)

        # for i in range(0, len(self.feature_list)):
        # df1 = self.dataset[self.sorted_entropy_list[:self.feature_list[i]]]

        # y1 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.qbc)
        # # y1 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.pool)
        # self.init(self.initial_point, self.query_number)
        #
        # y2 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.margin_pool)
        # self.init(self.initial_point, self.query_number)
        #
        # y3 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.entropy_pool)
        # self.init(self.initial_point, self.query_number)
        #
        # y4 = self.active_learn(self.df1, self.first_item_index_of_each_category, Method.qbc)
        # self.init(self.initial_point, self.query_number)
        #
        # self.plotter(y1, y2, y3, y4, self.query_number - 1, 15)


# al1 = BinaryAL(45, 150)
# al1.learnAndPlot()
#
# al1 = BinaryAL(75, 150)
# al1.learnAndPlot()

al1 = BinaryAL(10, 120)
# al1.learnAndPlot()

al2 = BinaryAL(10, 120)
# al2.learnAndPlot()

al3 = BinaryAL(10, 120)
# al3.learnAndPlot()

al4 = BinaryAL(10, 120)
al5 = BinaryAL(10, 120)
al6 = BinaryAL(10, 120)
# al4.learnAndPlot()

acc, pre, rec, fs = al1.learn_pool_proba(RandomForestClassifier(), uncertainty_sampling, 40)
dic = {
    'accuracy': acc,
    'precision': pre,
    'recall': rec,
    'f1 score': fs

}
al1.dumb_plotter_proba(dic, 5, 40)

acc, pre, rec, fs = al2.learn_pool_proba(RandomForestClassifier(), uncertainty_sampling, 30)
dic = {
    'accuracy': acc,
    'precision': pre,
    'recall': rec,
    'f1 score': fs

}
al1.dumb_plotter_proba(dic, 5, 30)

acc, pre, rec, fs = al3.learn_pool_proba(RandomForestClassifier(), uncertainty_sampling, 20)
dic = {
    'accuracy': acc,
    'precision': pre,
    'recall': rec,
    'f1 score': fs

}
al1.dumb_plotter_proba(dic, 5, 20)

acc, pre, rec, fs = al4.learn_pool_proba(RandomForestClassifier(), uncertainty_sampling, 10)
dic = {
    'accuracy': acc,
    'precision': pre,
    'recall': rec,
    'f1 score': fs

}
al1.dumb_plotter_proba(dic, 5, 10)

#
# dic = {
#     'Least confident (uncertainty sampling)' :  al1.learn_and_plot_pool(RandomForestClassifier(), uncertainty_sampling),
#     'Entropy sampling (uncertainty sampling)' :  al2.learn_and_plot_pool(RandomForestClassifier(), entropy_sampling),
#     'Margin sampling (uncertainty sampling)' :  al3.learn_and_plot_pool(RandomForestClassifier(), margin_sampling),
#     'Vote entropy sampling (query by committee)' :  al4.learn_and_plot_qbc(vote_entropy_sampling),
#     'Consensus entropy sampling (query by committee)' :  al5.learn_and_plot_qbc(consensus_entropy_sampling),
#     'Max disagreement sampling (query by committee)' :  al6.learn_and_plot_qbc(max_disagreement_sampling),
#     # 'max_std_sampling' :  al4.learn_and_plot_qbc(max_std_sampling),
# }


# al1.dumb_plotter_pool(dic, 5, 'recall')

