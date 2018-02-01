import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class TLP(Chain):  # Two Layer Perceptron

    def __init__(self, n_units, n_out):
        super(TLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        return self.l2(h1)


# Set up parameters
unit = 50
batch_size = 100
epoch = 100
out = 'result'

# Set up a neural network to train
model = L.Classifier(TLP(unit, 10))

# Set up an optimizer
optimizer = optimizers.SGD()
optimizer.setup(model)

# Load the MNIST dataset
train, test = datasets.get_mnist()
train_iter = iterators.SerialIterator(train, batch_size)
test_iter = iterators.SerialIterator(
    test, batch_size, repeat=False, shuffle=False)

# Set up a trainer
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

# Write logs
trainer.extend(extensions.LogReport())

# Print logs to stdout
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

# Run the training
trainer.run()
