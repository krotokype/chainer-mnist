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
from os import path

# Set up parameters
unit = 50
batch_size = 100
epoch = 100
frequency = epoch // 10
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

# Evaluate the model with the test dataset
trainer.extend(extensions.Evaluator(test_iter, model))

# Take snapshots
trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

# Write logs
trainer.extend(extensions.LogReport())

# Print logs to stdout
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

# Resume from a snapshot
if len(argv) > 1:
    serializers.load_npz(argv[1], trainer)

# Run the training
trainer.run()

# Take a model
serializers.save_npz(path.join(out, 'model.npz'), model)
