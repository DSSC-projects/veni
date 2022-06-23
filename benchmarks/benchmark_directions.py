import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import jax
from jax import jit, grad
import csv
import jax.numpy as jnp
from veni import MLP, ReLU, Softmax, Sequential, Linear
from veni.module import Module
from veni.utils import NumpyLoader, one_hot
from veni.optim import SGD, Adam, grad_fwd, NormalLikeSampler
from veni.functiontools import CrossEntropy
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
from tqdm import tqdm



PATH = "./"
# define hyperparameters
num_epochs = 10
batch_size = 24
n_targets = 10
key = jax.random.PRNGKey(111)
sampler = NormalLikeSampler()

# define network


class MLP(Module):
    def __init__(self):
        self.layers = Sequential([
            Linear(28*28, 1024, key),
            ReLU(),
            Linear(1024, 10, key),
            Softmax()
        ])

        self.params = self.layers.generate_parameters()
        # eliminate the bias

    def forward(self, x, params):
        return self.layers(x, params)


model = MLP()
params = model.params

# loss and accuracy


def loss(params, x, y):
    y_hat = model(x, params)
    return CrossEntropy(y, y_hat)


def accuracy(y, y_hat):
    model_predictions = jnp.argmax(y_hat, axis=1)
    return jnp.mean(y == model_predictions)


class tf(object):
    def __call__(self, pic):
        return np.array(np.ravel(pic), dtype=jnp.float32)/255


def evaluatePerf(gen):
    acc = 0
    count = 0
    for x, y in gen:
        y_hat = model(x, params)
        acc += accuracy(y, y_hat)*x.shape[0]
        count += x.shape[0]
    return acc / count

# Get dataset


# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=tf(), train=True)
training_generator = NumpyLoader(
    mnist_dataset, batch_size=batch_size, num_workers=0)

mnist_dataset = MNIST('/tmp/mnist/', download=True,
                      transform=tf(), train=False)
testing_generator = NumpyLoader(
    mnist_dataset, batch_size=batch_size, num_workers=0)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.train_data).reshape(
    len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(
    len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)



def grad_bwd(params, x, y, loss):
    grads = grad(loss)(params, x, y)
    return grads


def update(params, x, y, loss, optimizer, dirs=1, grad_type='fwd', sampler=None):
    if grad_type == 'fwd':
        if sampler is None:
            grads = grad_fwd(params, x, y, loss, dirs)
        else:
            grads = grad_fwd(params, x, y, loss, dirs, sampler)
    elif grad_type == 'bwd':
        grads = grad_bwd(params, x, y, loss)
    else:
        raise ValueError(
            f"Invalid grad_type, expected 'fwd' or 'bwd' got '{grad_type}'")

    return optimizer(params, grads)


# =========== HERE TEST START ========== #
model = MLP()
params = model.params

# ADAM test 1
print("ADAM test 1 start: lr = 1e-3")
optimizer = Adam(params, eta=1e-3)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

with open(PATH+'adam_1e-3.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(iteration, final_accuracy))

plt.plot(jnp.array(iteration), jnp.array(final_accuracy))
plt.savefig(PATH+'adam_1e-3.png')
plt.close()


# ADAM test
model = MLP()
params = model.params

print("ADAM test 2 start: lr = 1e-4")
optimizer = Adam(params, eta=1e-4)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

# ADAM test 3
model = MLP()
params = model.params

print("ADAM test 3 start: lr = 1e-5")
optimizer = Adam(params, eta=1e-5)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

with open(PATH+'adam_1e-5.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(iteration, final_accuracy))

plt.plot(jnp.array(iteration), jnp.array(final_accuracy))
plt.savefig(PATH+'adam_1e-5.png')
plt.close()

# SGD test
model = MLP()
params = model.params

print("SGD test 1 start: lr = 1e-3, momentum = 0, dampening = 0")
optimizer = SGD(params, eta=1e-3)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

with open(PATH+'sgd_1e-3-0-0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(iteration, final_accuracy))

plt.plot(jnp.array(iteration), jnp.array(final_accuracy))
plt.savefig(PATH+'sgd_1e-3-0-0.png')
plt.close()

# SGD test
model = MLP()
params = model.params

print("SGD test 2 start: lr = 1e-4, momentum = 0, dampening = 0")
optimizer = SGD(params, eta=1e-4)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

with open(PATH+'sgd_1e-4-0-0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(iteration, final_accuracy))

plt.plot(jnp.array(iteration), jnp.array(final_accuracy))
plt.savefig(PATH+'sgd_1e-4-0-0.png')
plt.close()

# SGD test
model = MLP()
params = model.params

print("SGD test 3 start: lr = 1e-5, momentum = 0, dampening = 0")
optimizer = SGD(params, eta=1e-5)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

with open(PATH+'sgd_1e-5-0-0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(iteration, final_accuracy))

plt.plot(jnp.array(iteration), jnp.array(final_accuracy))
plt.savefig(PATH+'sgd_1e-5-0-0.png')
plt.close()

# SGD test
model = MLP()
params = model.params

print("SGD test 4 start: lr = 1e-3, momentum = 0.9, dampening = 0")
optimizer = SGD(params, eta=1e-3, momentum=0.9)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

with open(PATH+'sgd_1e-3-09-0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(iteration, final_accuracy))

plt.plot(jnp.array(iteration), jnp.array(final_accuracy))
plt.savefig(PATH+'sgd_1e-3-0.9-0.png')
plt.close()

# SGD test 1
model = MLP()
params = model.params

print("SGD test 5 start: lr = 1e-4, momentum = 0.9, dampening = 0")
optimizer = SGD(params, eta=1e-4, momentum=0.9)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

with open(PATH+'sgd_1e-4-0.9-0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(iteration, final_accuracy))

plt.plot(jnp.array(iteration), jnp.array(final_accuracy))
plt.savefig(PATH+'sgd_1e-4-0.9-0.png')
plt.close()

# SGD test 1
model = MLP()
params = model.params

print("SGD test 6 start: lr = 1e-5, momentum = 0.9, dampening = 0")
optimizer = SGD(params, eta=1e-5, momentum=0.9)

final_accuracy = []
iteration = []
count = 0
for epoch in tqdm(range(num_epochs)):
    for x, y in training_generator:
        key = jax.random.split(key)
        one_hot_label = one_hot(y, n_targets)
        y_hat = model(x, params)

        # update parameters
        params = update(params, x, one_hot_label, loss, optimizer, dirs=1, grad_type='fwd', sampler=sampler)
        count += 1

    iteration.append(count + epoch)
    final_accuracy.append(float(evaluatePerf(testing_generator)))

with open(PATH+'sgd_1e-5-0.9-0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(iteration, final_accuracy))

plt.plot(jnp.array(iteration), jnp.array(final_accuracy))
plt.savefig(PATH+'sgd_1e-5-0.9-0.png')
plt.close()
