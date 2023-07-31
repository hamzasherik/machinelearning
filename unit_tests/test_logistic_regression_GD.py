import pandas as pd
import numpy as np

from unittest import TestCase
from logistic_regression_GD import LogisticRegressionGD

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

class TestLogisticRegressionGD(TestCase):

    def setUp(self):
        data = pd.read_csv(data_url).iloc[:99].values

        self.X = data[:,:4]
        self.y = data[:, 4]

        # first 49 elements of dataset are setosa, the next 50 are versicolor
        self.y_preprocessed_arr = np.concatenate((np.zeros(49, dtype=float), np.ones(50, dtype=float)))

    def test_init(self):
        logistic_regression_instance = LogisticRegressionGD(np.float64(0.01), np.int32(10), np.int32(1), np.float64(0.8))

        self.assertEqual(np.float64(0.01), logistic_regression_instance.eta)
        self.assertEqual(np.int32(10), logistic_regression_instance.epochs)
        self.assertEqual(np.int32(1), logistic_regression_instance.random_state)
        self.assertEqual(np.float64(0.8), logistic_regression_instance.threshold_accuracy)

    def test_preprocess(self):
        logistic_regression_instance = LogisticRegressionGD(np.float64(0.01), np.int32(10), np.int32(1), np.float64(0.8))

        X_train, X_test, y_train, y_test = logistic_regression_instance.preprocess(self.X, self.y)
        

        self.assertEqual(type(X_train), np.ndarray)
        self.assertEqual(type(X_test), np.ndarray)
        self.assertEqual(type(y_train), np.ndarray)
        self.assertEqual(type(y_test), np.ndarray)
    
        self.assertEqual(X_train.shape[0], 79)
        self.assertEqual(X_train.shape[1], 4)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(X_test.shape[1], 4)
        self.assertEqual(y_train.size, 79)
        self.assertEqual(y_test.size, 20)

    def test_lifecycle(self):
        logistic_regression_instance = LogisticRegressionGD(np.float64(0.01), np.int32(10), np.int32(1), np.float64(0.8))

        logistic_regression_instance.lifecycle(self.X, self.y)

        