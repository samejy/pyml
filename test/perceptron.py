
def simple_test():

    trainX = np.array([[1,1],[1,2],[2,1],[5,5],[6,5],[0,2],[1,0],[7,7]])
    trainy = np.array([0,0,0,1,1,0,0,1])

    test = np.array([[1,1],[7,8],[2,1]])

    expected = np.array([0, 1, 0])

    p = Perceptron(trainX, trainy)
    prediction = p.predict(test)
    assert(sum(abs(prediction - expected)) == 0, "Results do not match expected")

