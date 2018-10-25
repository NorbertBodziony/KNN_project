import itertools
import sys
import numpy as np
import operator
from scipy.spatial import distance
from collections import defaultdict
import pandas
from scipy.stats import pearsonr
from statistics import mode


def most_common(L: list):
    # get an iterable of (item, iterable) pairs
    if not isinstance(L, list):
        raise TypeError("wrong type")
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


class knn:
    def __init__(self, k: int, learning_data: np):
        if not isinstance(k, int):
            raise TypeError("k type is not int")

        self.data, self.label = np.array_split(learning_data, [4], axis=1)
        self.label = np.array(self.label).tolist()
        size = [i for i in range(0, self.data.shape[0])]
        if k > len(size):
            print("K is too big ")
            return -1
        else:
            self.k = k

    def predict(self, test_data: np):
        if test_data.shape[1] - 1 < self.data.shape[1] | (not isinstance(test_data, np.ndarray)):
            raise TypeError("Array too small or is not NdArray")

        distance_data = np.zeros(shape=(test_data.shape[0], self.data.shape[0]))


        data_test, label_test = np.array_split(test_data, [4], axis=1)
        for i in range(0, data_test.shape[0]):
            for j in range(0, self.data.shape[0]):
                distance_data[i][j] = distance.euclidean(data_test[i], self.data[j])
        size = [i for i in range(0, self.data.shape[0])]

        l = []
        ret = []
        for j in range(0, data_test.shape[0]):
            dict_distance_data = dict(zip(size, distance_data[j]))
            sort_dict_distance_data = list(sorted(dict_distance_data.items(), key=operator.itemgetter(1)))
            for i in range(0, self.k):
                l.append((self.label[sort_dict_distance_data[i][0]]))

            ret.append(most_common(l))

            l.clear()
        return ret
    def predict_pearson(self, test_data: np):
        if test_data.shape[1] - 1 < self.data.shape[1] | (not isinstance(test_data, np.ndarray)):
            raise TypeError("Array too small or is not NdArray")

        distance_data = np.zeros(shape=(test_data.shape[0], self.data.shape[0]))


        data_test, label_test = np.array_split(test_data, [4], axis=1)
        for i in range(0, data_test.shape[0]):
            for j in range(0, self.data.shape[0]):
                a,distance_data[i][j] = pearsonr(self.data[j], data_test[i])
        size = [i for i in range(0, self.data.shape[0])]

        l = []
        ret = []
        for j in range(0, data_test.shape[0]):
            dict_distance_data = dict(zip(size, distance_data[j]))
            sort_dict_distance_data = list(sorted(dict_distance_data.items(), key=operator.itemgetter(1)))
            for i in range(0, self.k):
                l.append((self.label[sort_dict_distance_data[i][0]]))

            ret.append(most_common(l))

            l.clear()
        return ret
    def predict_pearson(self, test_data: np):
        if test_data.shape[1] - 1 < self.data.shape[1] | (not isinstance(test_data, np.ndarray)):
            raise TypeError("Array too small or is not NdArray")

        distance_data = np.zeros(shape=(test_data.shape[0], self.data.shape[0]))


        data_test, label_test = np.array_split(test_data, [4], axis=1)
        for i in range(0, data_test.shape[0]):
            for j in range(0, self.data.shape[0]):
                a,distance_data[i][j] = pearsonr(self.data[j], data_test[i])
        size = [i for i in range(0, self.data.shape[0])]

        l = []
        ret = []
        for j in range(0, data_test.shape[0]):
            dict_distance_data = dict(zip(size, distance_data[j]))
            sort_dict_distance_data = list(sorted(dict_distance_data.items(), key=operator.itemgetter(1)))
            for i in range(0, self.k):
                l.append((self.label[sort_dict_distance_data[i][0]]))

            ret.append(most_common(l))

            l.clear()
        return ret
    def score(self, test_data: np):
        data_test, label_test = np.array_split(test_data, [4], axis=1)
        label_result = self.predict(test_data)
        if label_result == -1:
            return "k is too big"
        label_test = np.array(label_test).tolist()
        label_result = np.array(label_result).tolist()

        result = 0

        for i in range(0, len(label_result)):
            if label_result[i][0] == label_test[i][0]:
                result += 1
        return result, len(label_result)

    def score_pearson(self, test_data: np):
           data_test, label_test = np.array_split(test_data, [4], axis=1)
           label_result = self.predict_pearson(test_data)
           if label_result == -1:
               return "k is too big"
           label_test = np.array(label_test).tolist()
           label_result = np.array(label_result).tolist()

           result = 0

           for i in range(0, len(label_result)):
               if label_result[i][0] == label_test[i][0]:
                   result += 1
           return result, len(label_result)

try:
    learning_data = np.array(pandas.read_csv("iris.data.learning", header=None))
    test_data = np.array(pandas.read_csv("iris.data.test", header=None))
except:
    print("error in reading a file")

try:
    a = knn(10, learning_data)
    print(a.score_pearson(test_data))
except TypeError:
    print("Error in creating object")

