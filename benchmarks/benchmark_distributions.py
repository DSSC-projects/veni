from torchvision.datasets import MNIST
import numpy as np
import csv
import jax
import matplotlib.pyplot as plt
from jax import grad
from veni.net import Module, Sequential, Linear
from veni.function import Softmax, ReLU
from veni.utils import one_hot, NumpyLoader
from veni.functiontools import CrossEntropy
from veni.optim import Adam, grad_fwd, RademacherLikeSampler, TruncatedNormalLikeSampler, NormalLikeSampler, UniformLikeSampler
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm


key = jax.random.PRNGKey(10)
batch_size = 64
epochs = 30

# flatten and normalize


class tf(object):
    def __call__(self, pic):
        return (np.ravel(np.array(pic, dtype=jnp.float32)) / 255. - 0.5) * 2


training_dataset = MNIST('/tmp/mnist/', train=True,
                         download=True, transform=tf())
training_generator = NumpyLoader(training_dataset, batch_size=batch_size)


testing_dataset = MNIST('/tmp/mnist/', train=False,
                        download=True, transform=tf())
testing_generator = NumpyLoader(testing_dataset, batch_size=batch_size)


class MLP(Module):
    def __init__(self):
        self.layers = Sequential([
            Linear(28*28, 1024, jax.random.PRNGKey(10)),
            ReLU(),
            Linear(1024, 1024, key),
            ReLU(),
            Linear(1024, 10, key),
            Softmax()
        ])

        self.params = self.layers.generate_parameters()
        # eliminate the bias

    def forward(self, x, params):
        return self.layers(x, params)


def loss(params, x, y):
    y_hat = model(x, params)
    return CrossEntropy(y, y_hat)


def accuracy(y, y_hat):
    model_predictions = jnp.argmax(y_hat, axis=1)
    return jnp.mean(y == model_predictions)


def update(params, x, y, loss, optimizer,  grad_type='fwd', sampler=None):
    if grad_type == 'fwd':
        if sampler is None:
            grads = grad_fwd(params, x, y, loss, 1)
        else:
            grads = grad_fwd(params, x, y, loss, 1, sampler)
    else:
        raise ValueError(f"Invalid grad_type, expected 'fwd' got {grad_type}")

    return optimizer(params, grads)


# FIRST TEST = Normal Sampler
print("First test start")
model = MLP()
params = model.params
optimizer = Adam(params, eta=0.0005)
sampler = NormalLikeSampler()
res_loss1 = []
res_iter1 = []
count = 0
for epoch in range(epochs):
    running_loss = 0
    for i, (x, y) in enumerate(training_generator):
        key = jax.random.split(key)
        one_hot_label = one_hot(y, 10)
        running_loss += loss(params, x, one_hot_label)
        params = update(params, x, one_hot_label, loss,
                        optimizer, grad_type='fwd', sampler=sampler)
        count += 1
        if i % 100 == 99:
            res_loss1.append(running_loss/100)
            res_iter1.append(count)
            running_loss = 0

with open('normal_sampler.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(
        zip(res_loss1, res_iter1))

# SECOND TEST = Truncated Normal Sampler
print("Second test start")
model = MLP()
params = model.params
optimizer = Adam(params, eta=0.0005)
sampler = TruncatedNormalLikeSampler()
res_loss1 = []
res_iter1 = []
count = 0
for epoch in range(epochs):
    running_loss = 0
    for i, (x, y) in enumerate(training_generator):
        key = jax.random.split(key)
        one_hot_label = one_hot(y, 10)
        running_loss += loss(params, x, one_hot_label)
        params = update(params, x, one_hot_label, loss,
                        optimizer, grad_type='fwd', sampler=sampler)
        count += 1
        if i % 100 == 99:
            res_loss1.append(running_loss/100)
            res_iter1.append(count)
            running_loss = 0

with open('truncated_normal_sampler.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(
        zip(res_loss1, res_iter1))


# Third TEST = Truncated Normal Sampler
print("Third test start")
model = MLP()
params = model.params
optimizer = Adam(params, eta=0.0005)
sampler = RademacherLikeSampler()
res_loss1 = []
res_iter1 = []
count = 0
for epoch in range(epochs):
    running_loss = 0
    for i, (x, y) in enumerate(training_generator):
        key = jax.random.split(key)
        one_hot_label = one_hot(y, 10)
        running_loss += loss(params, x, one_hot_label)
        params = update(params, x, one_hot_label, loss,
                        optimizer, grad_type='fwd', sampler=sampler)
        count += 1
        if i % 100 == 99:
            res_loss1.append(running_loss/100)
            res_iter1.append(count)
            running_loss = 0

with open('rademacher_sampler.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(
        zip(res_loss1, res_iter1))


# Fourth TEST = Truncated Normal Sampler
print("Fourth test start")
model = MLP()
params = model.params
optimizer = Adam(params, eta=0.0005)
sampler = UniformLikeSampler()
res_loss1 = []
res_iter1 = []
count = 0
for epoch in range(epochs):
    running_loss = 0
    for i, (x, y) in enumerate(training_generator):
        key = jax.random.split(key)
        one_hot_label = one_hot(y, 10)
        running_loss += loss(params, x, one_hot_label)
        params = update(params, x, one_hot_label, loss,
                        optimizer, grad_type='fwd', sampler=sampler)
        count += 1
        if i % 100 == 99:
            res_loss1.append(running_loss/100)
            res_iter1.append(count)
            running_loss = 0

with open('uniform_sampler.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(
        zip(res_loss1, res_iter1))
