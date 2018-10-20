import sys
import numpy
import scipy
import pandas

class knn :
    def __init__(self,K,learning_data:list):
       self.K=K
       self.learning_data=learning_data

    def predict(self,test_data:list):
        pass

    def score(self,test_data:list):
        pass


learning_data=pandas.read_csv("iris.data.learning")
print(learning_data)