import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from model import TLP
from sys import argv
from sys import stdout

# Set up parameters
unit = 50
npz = argv[1]
index = int(argv[2])

# Set up a trained neural network
model = L.Classifier(TLP(unit, 10))
serializers.load_npz(npz, model)

# Load the MNIST dataset
_, test = datasets.get_mnist()

# Select data
x, _ = test[index % len(test)]

# Print selected data
def print2d(lines):
    for line in lines:
        for value in line:
            stdout.write('*' if value else ' ')
        print('')

print('-' * 28)
print2d(x.reshape(28, -1) > 0.5)
print('-' * 28)

# Predict
x = x[None, ...]  # (784,) -> (1, 784)
y = model.predictor(x)
r = int(y.data.argmax(axis=1))
print(r)
