import numpy as np

class Perceptron:


    def __init__(self, trainX, trainy):
        # trainX should be m x n numpy array of training data
        # m = number of examples, n = number of features per example
        # trainy should b m x 1 array of training classes (0 or 1)

        xshape = trainX.shape
        yshape = trainy.shape

        if len(xshape) != 2:
            error("trainX must be a 2 dimensional array")

        # etc...

        m = xshape[0]
        n = xshape[1]

        constant = np.ones(m).reshape((m,1))

        trainX2 = np.append(constant, trainX, 1)

        weights = np.zeros(n + 1)
        bestWeights = weights
        bestErr = 1e10
        countSinceLastImprovement = 0

        training = True

        while (training):

            updates = 0
            i = 0

            for example in trainX2:
                val = np.dot(weights, example)
                val = 1 if val > 0 else 0

                if abs(val - trainy[i]) > 0:
                    weights = weights + (trainy[i] - val) * example
                    updates = updates + 1

                i = i + 1

            if updates == 0:
                training = False
                self.weights = weights

            res = np.dot(trainX2, weights)
            res[res > 0] = 1
            res[res < 1] = 0
            err = sum(abs(res - trainy))
            countSinceLastImprovement = countSinceLastImprovement + 1

            if err < bestErr:
                bestWeights = weights
                bestErr = err
                countSinceLastImprovement = 0

            if countSinceLastImprovement > 20:
                training = False
                self.weights = bestWeights

    def predict(self, data):

        dshape = data.shape

        m = dshape[0]
        n = dshape[1]

        data2 = np.append(np.ones(m).reshape((m,1)), data, 1)

        res = np.dot(data2, self.weights)

        res[res > 0] = 1
        res[res < 1] = 0
        return res


