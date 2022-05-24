#! /usr/bin/python3

import os, sys
sys.path.append('../')
import jax 
import jax.numpy as jnp
from jax import grad, jvp
jax.config.update('jax_platform_name', 'cpu')

from jax_forward.net import Module, Sequential, Linear
from jax_forward.function import Sigmoid
from jax_forward.utils import one_hot, NumpyLoader, FlattenAndCast
from jax_forward.functiontools import CrossEntropy
from torchvision.datasets import MNIST
import numpy as np


dspath = '../../datasets'
batch_size = 128
num_epochs = 3
n_targets = 10
step_size = 1e-4

key = jax.random.PRNGKey(1)

class tf(object):
    def __call__(self, pic):
        return ( np.ravel(np.array(pic, dtype=jnp.float32)) / 255. - 0.5 ) * 2

print(f"Loading into/from path {dspath}")
train_dataset = MNIST(dspath, train = True, download= True, transform= tf())
train_generator = NumpyLoader(train_dataset, batch_size= 128 )


test_dataset = MNIST(dspath, train = False,download= True, transform= tf() )
test_generator = NumpyLoader(test_dataset, batch_size= 128 )
print("Loaded")


class LogisticRegressor(Module):
    def __init__(self):
        self.layers = Sequential([
            Linear(28*28, 10, key),
            Sigmoid()
        ])

        self.params = self.layers.generate_parameters()
        #eliminate the bias
    
    def forward(self,x,params):
        return self.layers(x,params)

model = LogisticRegressor()
params = model.params

def loss(params, x, y):
    y_hat = model(x,params)

    l1 = y*jnp.log(y_hat)
    l2 = (1-y)*jnp.log(1 - y_hat)
    l = -jnp.sum(l1 + l2)
    return l

def accuracy(y,y_hat):
    model_predictions = jnp.argmax(y_hat, axis= 1)
    return jnp.mean(y == model_predictions)

def update_bwd(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

for epoch in range(num_epochs):
    running_loss = 0
    for i, (image, label) in enumerate(train_generator):
        one_hot_label = one_hot(label, n_targets)
        params = update_bwd(params, image, one_hot_label)
        # loss info
        loss_item =  loss(params, image, one_hot_label)
        running_loss = running_loss + loss_item
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0  


#model assessment
acc = 0
count = 0
for x,y in train_generator:
    y_hat = model(x,params)
    acc += accuracy(y,y_hat)*x.shape[0]
    count += x.shape[0]

print(f"Training set accuracy {acc/count}")


acc = 0
count = 0
for x,y in test_generator:
    y_hat = model(x,params)
    acc += accuracy(y,y_hat)*x.shape[0]
    count += x.shape[0]

print(f"Test set accuracy {acc/count}")

