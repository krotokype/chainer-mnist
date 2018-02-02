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
