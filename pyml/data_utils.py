import numpy as np

def validate_inputs(trainX, trainy):
    xshape = trainX.shape
    yshape = trainy.shape

    if len(xshape) != 2:
        raise Exception("trainX must be a 2 dimensional array")

    m = xshape[0]
    n = xshape[1]

    if yshape[0] != m:
        raise Exception("trainy must contain same number of values as examples in trainX")

    return (m,n)


def prepend_constant_feature_value(X, nExamples, value):
    constant = np.ones(nExamples).reshape((nExamples,1)) * value
    result = np.append(constant, X, 1)
    return result
