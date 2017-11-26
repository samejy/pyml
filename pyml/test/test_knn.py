from unittest import TestCase
from pyml import knn
import numpy as np

class KnnTests(TestCase):

    def test_simple(self):

        trainX = np.array([[1,1],[1,2],[2,1],[5,5],[6,5],[0,2],[1,0],[7,7]])
        trainy = np.array([0,0,0,1,1,0,0,1])

        test = np.array([[1,1],[7,8],[2,1]])

        expected = np.array([0, 1, 0])

        knb = knn.KNearestNeighbours(trainX, trainy, 3)
        prediction = knb.predict(test)

        self.assertTrue(sum(abs(prediction - expected)) == 0)
