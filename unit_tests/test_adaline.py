from unittest import TestCase
import numpy as np
import pandas as pd
from adaline import Adaline

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

class TestAdaline(TestCase):

    def setUp(self):
        data = pd.read_csv(data_url).iloc[:99].values

        self.X = data[:,:4]
        self.y = data[:, 4]

        # first 49 elements of dataset are setosa, the next 50 are versicolor
        self.y_preprocessed_arr = np.concatenate((np.zeros(49, dtype=float), np.ones(50, dtype=float)))

    def test_init(self):
        adaline_instance = Adaline(eta=0.1, epochs=10)

        self.assertEqual(adaline_instance.eta, 0.1)
        self.assertEqual(adaline_instance.epochs, 10)
        self.assertEqual(adaline_instance.random_state, 1)
        self.assertEqual(adaline_instance.threshold_accuracy, 0.8)

    def test_preprocess(self):
        adaline_instance = Adaline(eta=0.1, epochs=10)

        X_preprocessed, y_preprocessed = adaline_instance.preprocess(self.X, self.y)

        np.testing.assert_array_almost_equal(X_preprocessed, self.X)
        np.testing.assert_array_almost_equal(y_preprocessed, self.y_preprocessed_arr)

    def test_lifecycle(self):
        adaline_instance = Adaline(eta=0.001, epochs=10)

        adaline_instance.lifecycle(self.X, self.y)
