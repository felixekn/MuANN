import numpy as np
import itertools


def BP_learn (trainInput, trainOutput, testInput, testOutput, networkLayers,
              slope, eta, n, tol, weights, batch, alpha, decay):
    errorTraining = []
    errorTest = []
    totalSteps = []
    errorHistory = []
    previousWeightChanges = copy_cell_structure(weights)

    # Randomizing training sets
    d = trainInput.shape

    if batch:
        randPerm = np.random.permutation(d[0])
    else:
        randPerm = np.random.randint(0,d[0]-1)

    # learning loop
    for i in range(n):

        newEta = tabularLearning(eta, i, n)

        y = layer_outputs(weights, trainInput[randPerm,:], slope, batch)
        deltas = feedback_errors(weights, trainOutput[randPerm,:], y, batch)
        weights, weightChanges = update_weights(weights, newEta, deltas, trainInput[randPerm,:],
                                                 y, alpha, previousWeightChanges, batch)

        weights[0] = (1-decay) * weights[0]
        weights[1] = (1-decay) * weights[1]

        previousWeightChanges = weightChanges

        if i % 600 == 0 or i == n - 1:
            recallTraining = BP_recall(weights, trainInput, slope)
            recallTest = BP_recall(weights, testInput, slope)

            thresholdedRecalTraining = threshold(recallTraining)
            thresholdedRecallTest = threshold(recallTest)

            error = [percent_identity(trainOutput, thresholdedRecalTraining),
                     percent_identity(testOutput, thresholdedRecallTest)]

        errorTraining.append(error[0])
        errorTest.append(error[1])

        if batch:
            totalSteps.append(i * trainInput.size)
        else:
            totalSteps.append(i)

        if error[0] >= tol:
            print("Gradient Search Terminated ===>>> Hit Rate >= %s")%(error)
            if batch:
                print('Number of Iterations = %s')%((i+1) * trainInput.size)
            else:
                print('Number of Iterations = %s')%(i+1)
            break

        # Randomizing training sets
        if batch:
            randPerm = np.random.permutation(d[0])
        else:
            randPerm = np.random.randint(0,d[0]-1)

    if error[0] < tol:
        print('Failed to reach Hit Rate >= %s: Hit Rate = %s')%(tol, error)
        if batch:
            print('Number of Iterations = %s')%((i+1) * trainInput.size)
        else:
            print('Number of Iterations = %s')%(i+1)

    errorHistory = [errorTraining, errorTest]

    return weights, totalSteps, errorHistory


# Initializes weight matrices (correct)
def initialize_weights(network, scaling):
    NUM_MATRICIES = len(network) - 1
    weights = []
    weightRange = 0.5

    for i in range(NUM_MATRICIES):
        # the "+1" in "Network[i]+1" adds the bias weights
        weights.append(np.random.rand(network[i]+1, network[i+1])*0.4 - 0.2)

    return weights


# Calculates outputs from hidden and output layer (correct)
def layer_outputs(weights, x, slope, batch, tanhScaler = 1):
    if not batch: x = np.array([x])
    d = x.shape
    dw = weights[0].shape
    dy = weights[1].shape
    hiddenLayerYMatrix = np.zeros((d[0], dw[1]))
    outputLayerYMatrix = np.zeros((d[0], dy[1]))

    # 1.7159

    for i in range(d[0]):
        # tanh(slope * f) where f = input vector * input weight matrix + bias weights
        hiddenLayerYMatrix[i,:] = tanhScaler * np.tanh(slope * (np.dot(x[i,:], weights[0][0:-1,:]) + weights[0][-1,:]))

        # tanh(slope * f) where f = (hidden layer outputs) * (output weight matrix) + bias weights
        outputLayerYMatrix[i,:] = tanhScaler * np.tanh(slope * (np.dot(hiddenLayerYMatrix[i,:], weights[1][0:-1,:]) + weights[1][-1,:]))

    return np.append(hiddenLayerYMatrix, outputLayerYMatrix, axis = -1)


# Calculates output feedback errors (correct)
def feedback_errors(weights, x, y, batch):
    if not batch: x = np.array([x])

    dy = y.shape
    dx = x.shape

    outputLayerErrorMatrix = np.zeros((dx[0], dx[1]))
    hiddenLayerErrorMatrix = np.zeros((dy[0], dy[1]-dx[1]))
    for i in range(dx[0]):
        outputLayerErrorMatrix[i,:] = np.multiply(x[i,:] - y[i,dy[1]-dx[1]:], 1 - np.square(y[i,dy[1]-dx[1]:]))
        hiddenLayerErrorMatrix[i,:] = np.multiply(np.dot(outputLayerErrorMatrix[i,:], weights[1][0:-1,:].T), 1 - np.square(y[i,0:-dx[1]])) #correct

    return np.append(hiddenLayerErrorMatrix, outputLayerErrorMatrix, axis = -1)


# Updates weights from feedback errors with momentum (correct)
def update_weights(weights, eta, deltas, x, y, alpha, previousWeightChanges, batch):
    if not batch: x = np.array([x])

    dy = y.shape
    mw, nw = weights[0].shape
    mw2, nw2 = weights[1].shape

    weightChanges = [np.zeros((mw, nw)), np.zeros((mw2, mw2))]
    newWeights = [np.zeros((mw, nw)), np.zeros((mw2, mw2))]

    hiddenWeightMatrix = np.zeros((mw, nw))
    outputLayerWeightMatrix = np.zeros((mw2, nw2))

    for i in range(dy[0]):
        hiddenWeightMatrix[0:-1,:] = hiddenWeightMatrix[0:-1,:] + eta * np.dot(np.matrix(x[i,:]).T, np.matrix(deltas[i,0:-nw2]))
        hiddenWeightMatrix[-1,:] = hiddenWeightMatrix[-1,:] + eta * deltas[i,0:-nw2] * 1

        outputLayerWeightMatrix[0:-1,:] = outputLayerWeightMatrix[0:-1,:] + eta * np.dot(np.matrix(y[i,0:-nw2]).T, np.matrix(deltas[i,-nw2:]))
        outputLayerWeightMatrix[-1,:] = outputLayerWeightMatrix[-1,:] + eta * deltas[i,-nw2:] * 1

    weightChanges[0] = hiddenWeightMatrix + alpha * previousWeightChanges[0]
    weightChanges[1] = outputLayerWeightMatrix + alpha * previousWeightChanges[1]

    newWeights[0] = weights[0] + weightChanges[0]
    newWeights[1] = weights[1] + weightChanges[1]

    return newWeights, weightChanges


def BP_recall(weights, x, slope):
    m, n = weights[1].shape
    y = layer_outputs(weights, x, slope, batch = True)
    recall = y[:,-n:]

    return recall


# Copies matrixCorrect
def copy_cell_structure(inputCell):
    copyedCell = []
    for cell in inputCell:
        d = cell.shape
        copyedCell.append(np.zeros(d))

    return copyedCell

# probably okay
def percent_identity(desiredOutput, recalledOutput):
    m, n = recalledOutput.shape
    equalityVector = np.zeros((m,1))

    for i in range(m):
        rowEquality = int(np.array_equal(desiredOutput[i,:], recalledOutput[i,:]))
        equalityVector[i,0] = rowEquality

    return np.sum(equalityVector)/float(m)

# categorize frequency inputs into n number of bins between two values.
#   maxiumum number of bins is 16.
def categorize(x, minVal, maxVal, nbins = 10):
    categorizedInput = []
    binnedInput = []

    # threshold decides what category x goes into
    threshold = (maxVal - minVal)/nbins

    # 16 total possible categories. ex [-1, -1, -1, 1]
    categories = [list(c) for c in itertools.product([-0.9,0.9],repeat=4)]

    for val in x:
        binNum = int(np.minimum(np.ceil(val / float(threshold))+1, nbins))
        categorizedInput.append(categories[binNum-1])
        binnedInput.append(binNum)

    return np.array(categorizedInput), np.array(binnedInput)


# converts the categorized output data (16 possible categories) into bin values
#   spanning [0,16] where 0 means not correctly binned.
def categoriesToBins(x, nbins = 10):
    binnedValues = []

    # 16 total possible categories. ex [-1, -1, -1, 1]
    categories = [list(c) for c in itertools.product([-0.9,0.9],repeat=4)]

    for val in x:
        binNum = categories.index(list(val)) + 1
        if binNum > nbins:
            binNum = 0
        binnedValues.append(binNum)

    return np.array(binnedValues)


def threshold(x):
    m, n = x.shape
    threshold = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if x[i,j] < 0:
                threshold[i,j] = -0.9
            elif x[i,j] >= 0:
                threshold[i,j] = 0.9

    return threshold

def tabularLearning(learningRate, k, totalSteps):
    newRate = 0
    if (k < totalSteps/4.):
        newRate = learningRate
    elif (k < totalSteps * 1/2.):
        newRate = learningRate/  2.
    elif (k < totalSteps * 2/3.):
        newRate = learningRate / 5.
    elif (k >= totalSteps * 2/3.):
        newRate = learningRate / 15.
    return newRate
