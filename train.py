import numpy as np
import pandas as pd

np.random.seed(1)

def nonlin(x, deriv=False):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def init_weights(*layer):
	result = []
	for i in range(len(layer)-1):
		result = result + [2 * np.random.rand(layer[i],layer[i+1]) - 1]
	return result

def forward_prop(weights, input):
    currentActivations = [input]
    for synapses in weights:
        currentActivations = currentActivations + [nonlin(np.dot(currentActivations[-1],synapses))]

    return currentActivations

def backward_prop(activations, weights, output):
    predicterror = output - activations[-1]
    errors = [predicterror]
    deltas = [errors[0] * nonlin(activations[-1], deriv=True)]
    for layer in range(len(weights))[1:][::-1]:
        errors = errors + [deltas[-1].dot(weights[layer].T)]
        deltas = deltas + [errors[-1] * nonlin(activations[layer], deriv=True)]

    return deltas[::-1], np.mean(np.abs(predicterror))

def update_weights(activations, weights, deltas, lrate):
    for synapses in range(len(deltas)):
        weights[synapses] += activations[synapses].T.dot(deltas[synapses]) * lrate

def predict(pred):
    if(pred < 0.5):
        return 0
    return 1

def train(weights, input, output, lrate, epochs):
    for x in range(epochs):
        activations = forward_prop(weights, input)
        deltas, error = backward_prop(activations, weights, output)
        if (x% 10000) == 0:
            print "Epoch: " + str(x) + " | Error: " + str(error)
        update_weights(activations, weights, deltas, lrate)

def test(weights, input, output):
    predictions = forward_prop(weights, input)[-1]
    return float([v[0] == predict(v[1]) for v in zip(output, predictions)].count(True)) / len(output)


def kfoldcrossval(input, output, k, lrate, epochs, layers):
    folds_input = np.array_split(input, k)
    folds_output = np.array_split(output, k)
    errors = []

    for i in range(k):
        weights = init_weights(*layers)

        test_input = folds_input[i]
        test_output = folds_output[i]

        train_input = np.concatenate(np.delete(folds_input, i, 0))
        train_output = np.concatenate(np.delete(folds_output, i, 0))

        train(weights, train_input, train_output, lrate, epochs)
        result = test(weights, test_input, test_output)
        errors += [result]
        print("fold: " + str(i) + ", success rate: " + str(result))

    return errors

df =  pd.read_csv("data/processed_train.csv")

X = df.values[:,1:]
Y = df['Survived'].values[np.newaxis].T

input = X
output = Y
"""
input = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

output = np.array([[0],
			[1],
			[1],
			[0]])
"""
layers = [len(input[0]), 12, len(output[0])]
kfoldcrossval(input, output, 2, 0.01, 60000, layers)
