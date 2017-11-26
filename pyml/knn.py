import numpy as np
import pyml.data_utils as du

class KNearestNeighbours:

    trainX = []
    trainy = []
    k = 0
    miny = 0
    maxy = 0

    def __init__(self, trainX, trainy, k, t='classification'):
        self.trainX = trainX
        self.trainy = trainy
        self.miny = trainy.min()
        self.maxy = trainy.max()
        self.k = k
        self.t = t

    def predict(self, data):
        results = list()
        for row in data:
            # calculate distance to each example instance
            distances = [self.distance(row, s) for s in self.trainX]
            # create a list of tuples of (distance,index)
            inds = list(zip(distances, range(len(distances))))

            # sort indexes by distance to current example
            inds.sort(key = lambda t: t[0])
            # create a list of tuples of (count,class) with initial count 0
            res = [(0,i) for i in range(self.maxy + 1 - self.miny)]

            # get the class of the first k instances and
            # increment the result count for that class
            for i in range(self.k):
                cl = self.trainy[inds[i][1]]
                cur = res[cl - self.miny]
                res[cl - self.miny] = (cur[0]+1,cur[1])

            # sort by count
            res.sort(key = lambda t: t[0])
            # take the class of the largest count
            results.append(res[len(res) - 1][1])
        return results

    # if one vector is shorter than the other, only the elements up
    # to the length of the shorter vector are compared
    def distance(self, a, b):
        min_len = min(len(a), len(b))
        return sum([abs(ai - bi) for (ai, bi) in zip(a, b)]) / min_len


