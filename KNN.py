import itertools
import sys
import numpy as np
import operator
from scipy.spatial import distance
from collections import defaultdict
import pandas
from statistics import mode

def most_common(L):
  # get an iterable of (item, iterable) pairs
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

class knn :
    def __init__(self,K,learning_data):
       self.K=K
       self.data,self.label=np.array_split(learning_data,[4],axis=1)
       self.label=np.array(self.label).tolist()
    def predict(self, test_data):
        distance_data=np.zeros(shape=(14,134))

        size=[i for i in range(0,134)]

        data_test, label_test = np.array_split(test_data, [4], axis=1)

        for i in range(0,data_test.shape[0]):
            for j in range(0,self.data.shape[0]):

                distance_data[i][j]=distance.euclidean(data_test[i],self.data[j])


        l=[]
        ret=[]
        for j in range(0,data_test.shape[0]):
            dict_distance_data = dict(zip(size, distance_data[j]))
            sort_dict_distance_data = list(sorted(dict_distance_data.items(), key=operator.itemgetter(1)))
            for i in range(0,self.K):
                l.append((self.label[sort_dict_distance_data[i][0]]))

            ret.append(most_common(l))

            l.clear()
        return ret
    def score(self,test_data):
        data_test, label_test = np.array_split(test_data, [4], axis=1)
        label_result=self.predict(test_data)
        label_test=np.array(label_test).tolist()
        label_result=np.array(label_result).tolist()

        result=0

        for i in range(0,len(label_result)):
            if(label_result[i][0]==label_test[i][0]):
                    result+=1
        return result/len(label_result)

learning_data=np.array(pandas.read_csv("iris.data.learning"))
test_data=np.array(pandas.read_csv("iris.data.test"))
a=knn(5,learning_data)

print(a.score(test_data))