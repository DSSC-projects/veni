import jax 
import jax.numpy as jnp
from veni.net import *
from veni.functiontools import *
from veni.function import *
from veni.utils import * 
from torchvision.datasets import MNIST

import os, sys

jax.config.update('jax_platform_name','cpu')

dspath = '../datasets'
batch_size = 24
num_epochs = 5 
n_targets = 10
step_size = 1e-4


key = jax.random.PRNGKey(111111)

logging_freq = 200

class tf(object):
    def __call__(self, pic):
        return ( np.array(np.ravel(pic), dtype=jnp.float32)/ 255. - 0.5 ) * 2

print(f"Loading into/from path {dspath}")
train_dataset = MNIST(dspath, train = True, download= True, transform= tf())

n_train = train_dataset.data.shape[0]
train_generator = NumpyLoader(train_dataset, batch_size= batch_size )


test_dataset = MNIST(dspath, train = False,download= True, transform= tf() )

n_test = train_dataset.data.shape[0]
test_generator = NumpyLoader(test_dataset, batch_size= batch_size )
print("Loaded")



class LogisticRegressor(Module):
    def __init__(self):
        self.layers = Sequential([
            Linear(28*28, 10, key),
            Sigmoid()
        ])

        self.params = self.layers.generate_parameters()
    
    def forward(self,x,params):
        return self.layers(x,params)


def logisticLoss(y,y_hat):
    l1 = y*jnp.log(y_hat)
    l2 = (1-y)*jnp.log(1 - y_hat)
    l = -jnp.sum(l1 + l2)
    return l/y.shape[0]

def sample_random_direction(params, normalize = False):
    global key
    v_shaped = []
    for w, b in params:
        key, subkey = random.split(key)
        v_w = random.normal(key, shape=w.shape)
        v_b = random.normal(subkey, shape=b.shape)
        if normalize:
            v_shaped.append((v_w/jnp.linalg.norm(v_w), v_b/jnp.linalg.norm(v_b)))
        else:
            v_shaped.append((v_w,v_b))
    return v_shaped


def update_params_list(p,g,scale = 1.):
    return [ (w - dw*scale, b - db*scale) for (w,b), (dw,db) in zip(p,g)]

def accuracy(y,y_hat):
    n = y.shape[0]
    y = jnp.argmax(y, axis = 1)
    y_hat = jnp.argmax(y_hat, axis = 1)

    return jnp.sum(y == y_hat)/n

def train_loop(model, train_generator, validation_generator, loss, nepochs, batchIteration, lr, log_every_n_batches):  
    params = model.params
    global key
    train_loss = [] 
    validation_loss = [] 

    train_acc = []
    validation_acc = []

    def lossFn(params, x, y):
        y_hat = model(x,params)
        return loss(y,y_hat)

    for epoch in range(nepochs):
        print(f"--- Epoch : {epoch} ---")
        l = 0
        a = 0
        print(f"\t Training")
        for i, (x,y) in enumerate(train_generator):
            key = jax.random.split(key)
            one_hot_y = one_hot(y,n_targets)
            params = batchIteration(params,x,one_hot_y,lossFn, lr)
            y_hat = model(x, params)
            l += loss(one_hot_y,y_hat)
            a += accuracy(one_hot_y,y_hat)

            if i % log_every_n_batches == (log_every_n_batches - 1):
                train_loss.append([epoch,i,float(l)/log_every_n_batches])
                train_acc.append([epoch,i,float(a)/log_every_n_batches])
                l = 0
                a = 0

                print(f"\t l: {train_loss[-1]}")
                print(f"\t a: {train_acc[-1]}")

        l = 0
        a = 0
        
        for i, (x,y) in enumerate(validation_generator):
            one_hot_y = one_hot(y,n_targets)
            y_hat = model(x,params)
            l += loss(one_hot_y,y_hat)
            a += accuracy(one_hot_y,y_hat)

        validation_loss.append([epoch,i,float(l)/ i])
        validation_acc.append([epoch,i,float(a)/ i])
        print(f"\t Validation")
        print(f"\t l: {validation_loss[-1]}")
        print(f"\t a: {validation_acc[-1]}")



    return [train_loss, train_acc, validation_loss, validation_acc]

def bwd_train_batch(params, x, y, lossFn, lr):
    grad = jax.grad(lossFn)(params,x,y)
    params = update_params_list(params,grad, lr)
    return params

def single_fwd_train_batch(params,x,y,lossFn,lr):
    global key
    key = jax.random.split(key)
    v = sample_random_direction(params, normalize=False)
    _, j = jax.jvp(lambda p: lossFn(p,x,y), (params, ), (v,))
    params = update_params_list(params, v, j*lr)
    return params


def multiple_fwd_train_batch(params,x,y,lossFn,lr, dirs ):
    global key
    for i in range(dirs):
        key, _ = jax.random.split(key)
        v = sample_random_direction(params, normalize= False)
        _, j = jax.jvp(lambda p: lossFn(p, x, y), (params, ), (v,))
        grad_now = [(j * dw, j * db) for (dw, db) in v]

        if i == 0:
            grad = grad_now

        elif i > 0:
            grad = [(dw + dw_prev, db + db_prev)
                           for (dw, db), (dw_prev, db_prev) in zip(grad_now, grad)]


    if dirs != 1:
        grad = [(dw / dirs, db / dirs) for (dw, db) in grad]

    params = update_params_list(params,grad,lr) 
    return params

import pickle

os.makedirs('logs', exist_ok=True)

num_epochs = 18
m = LogisticRegressor()
vals = train_loop(m,train_generator, test_generator, logisticLoss, num_epochs, bwd_train_batch, 1e-4, 200)
with open(f"logs/r1_b",'wb') as f:
    pickle.dump(vals, f)

for d in [1, 10, 50, 100]:

    print(f"Training with {d} directions \n \n")

    d_fwd_train_batch = lambda params, x, y, lossFn, lr: multiple_fwd_train_batch(params, x, y, lossFn, lr, d)
    
    m = LogisticRegressor()
    vals = train_loop(m,train_generator, test_generator, logisticLoss, num_epochs, d_fwd_train_batch, 1e-4, 200)
    with open(f"logs/r1_{d}",'wb') as f:
        pickle.dump(vals, f)
