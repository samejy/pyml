from unittest import TestCase
from pyml import linear_regression
import numpy as np

class LinearRegressionTests(TestCase):

    def test_simple(self):

        trainX = np.array([[2,2],[3,3]])
        trainy = np.array([2,3])

        test = np.array([[4,4]])

        expected = np.array([4])

        l = linear_regression.LinearRegression(trainX, trainy)

        prediction = l.predict(test)

        self.assertTrue(sum(abs(prediction - expected)) < 1e-10)

    def test2(self):

        trainX = np.array([[2,2,-3],[3,3,-4]])
        trainy = np.array([5,6])

        test = np.array([[2.5,2.5,-3.5],[1,1,-2]])

        expected = np.array([5.5,4])

        l = linear_regression.LinearRegression(trainX, trainy)

        prediction = l.predict(test)

        self.assertTrue(sum(abs(prediction - expected)) < 1e-10)

