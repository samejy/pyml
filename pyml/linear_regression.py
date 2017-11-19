import numpy as np
import pyml.data_utils as du

class LinearRegression:

    weights = []
    feature_count = 0

    def __init__(self, trainX, trainy):
        (m,n) = du.validate_inputs(trainX, trainy)

        trainX2 = du.prepend_constant_feature_value(trainX, m, 1)

        xt = trainX2.T
        xtxinv = np.linalg.pinv(np.dot(xt, trainX2))
        xty = np.dot(xt, trainy)

        self.weights = np.dot(xtxinv, xty)
        self.feature_count = len(self.weights)
        return

    def predict(self, data):
        m = data.shape[0]

        data2 = du.prepend_constant_feature_value(data, m, 1)

        return np.dot(data2, self.weights)
