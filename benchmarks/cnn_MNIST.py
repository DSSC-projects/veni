#! /usr/bin/python3

import os, sys
sys.path.append('../')
import jax 
import jax.numpy as jnp
from jax import grad, jvp
jax.config.update('jax_platform_name', 'cpu')

from jax_forward.net import Module, Sequential, Linear, Conv2D, MaxPool2D, Flatten
from jax_forward.function import Sigmoid, ReLU, Softmax
from jax_forward.utils import one_hot, NumpyLoader, FlattenAndCast, _get_vector
from jax_forward.functiontools import CrossEntropy
from torchvision.datasets import MNIST
import numpy as np


dspath = '../../datasets'
batch_size = 64
num_epochs = 10
n_targets = 10
step_size = 2e-4

key = jax.random.PRNGKey(10)

#normalize
class tf(object):
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32).reshape([1, 28, 28]) / 255.

print(f"Loading into/from path {dspath}")
train_dataset = MNIST(dspath, train = True, download= True, transform= tf())
train_generator = NumpyLoader(train_dataset, batch_size= batch_size )


test_dataset = MNIST(dspath, train = False,download= True, transform= tf() )
test_generator = NumpyLoader(test_dataset, batch_size= batch_size )
print("Loaded")


class LogisticRegressor(Module):
    def __init__(self):
        self.layers = Sequential([
            Conv2D(1,64,3,1,'VALID',key),
            ReLU(),
            Conv2D(64,64,3,1,'VALID', key),
            ReLU(),
            MaxPool2D(2),
            Conv2D(64,64,3,1,'VALID', key),
            ReLU(),
            Conv2D(64,64,3,1,'VALID', key),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Linear(1024, 1024, key),
            ReLU(),
            Linear(1024, 10, key),
            Softmax()
        ])

        self.params = self.layers.generate_parameters()
        #eliminate the bias
    
    def forward(self,x,params):
        return self.layers(x,params)

model = LogisticRegressor()
params = model.params

def loss(params, x, y):
    y_hat = model(x,params)
    return CrossEntropy(y, y_hat)

def accuracy(y,y_hat):
    model_predictions = jnp.argmax(y_hat, axis= 1)
    return jnp.mean(y == model_predictions)

def update_bwd(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

print(f"Backward training")

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
def get_vector(params):
    v_shaped = []
    for w, b in params:
        v_w = jnp.array(np.random.normal(0,1, w.shape))
        v_b = jnp.array(np.random.normal(0,1, b.shape))
        v_shaped.append((v_w, v_b))
    return v_shaped

def update_fwd(params, x, y):
    v = _get_vector(key,params)
    _ , proj = jvp(lambda p: loss(p,x,y), (params, ), (v,) )
    return [(w - step_size * proj * dw, b - step_size * proj * db)
          for (w, b), (dw, db) in zip(params, v)]

print(f"Forward training")
model = LogisticRegressor()
params = model.params

for epoch in range(num_epochs):
    running_loss = 0
    for i, (image, label) in enumerate(train_generator):
        key = jax.random.split(key)
        one_hot_label = one_hot(label, n_targets)
        params = update_fwd(params, image, one_hot_label)
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