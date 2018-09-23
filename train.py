import numpy as np
import pandas as pd

np.random.seed(1)

def nonlin(x, deriv=False):
    if(deriv):
        return x * (x - 1)
    return 1 / (1 + np.exp(-x))

def init_weights(*layer):
	result = []
	for i in range(len(layer)-1):
		result = result + [2 * np.random.rand(layer[i],layer[i+1]) - 1]
	return result

def forward_prop(input, weights):
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
    for synapses in range(len(deltas))[:1]:
        weights[synapses] += activations[synapses].T.dot(deltas[synapses]) * lrate
    return weights

def predict(input, weights):
    return forward_prop(input, output, weights)[-1]


df =  pd.read_csv("data/processed_train.csv")

X = df.values[:,1:]
Y = df['Survived'].values[np.newaxis].T

weights = init_weights(len(X[0]),5,6,9,1)

input = X
output = Y



for x in range(60000):
    activations = forward_prop(input, weights)
    deltas, error = backward_prop(activations, weights, output)
    if (x% 10000) == 0:
        print "Error:" + str(error)
    update_weights(activations, weights, deltas, 0.01)

total = 714
true = 0

for x in range(len(input)):
