import unittest
import KNN
import numpy as np
import pandas


class test_KNN(unittest.TestCase):

    def setUp(self):
        try:
            self.learning_data = np.array(pandas.read_csv("iris.data.learning"))
            self.test_data = np.array(pandas.read_csv("iris.data.test"))
        except:
            print("error in reading a file")

    def test_Obcjet(self):

        self.assertRaises(TypeError, lambda: KNN.knn(3.2, self.learning_data))
        # self.assertIsInstance(a.K,int)

    def test_predict(self):
        data_test, label_test = np.array_split(self.test_data, [4], axis=1)
        self.assertRaises(TypeError,  KNN.knn(3, data_test))
        # self.assertIsInstance(a.K,int)

    if __name__ == '__main__':
        unittest.main()
