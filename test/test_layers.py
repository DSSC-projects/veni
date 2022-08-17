import sys
import jax 
jax.config.update('jax_platform_name','cpu')
sys.path.append(('../','.'))

from veni.net import *
from veni.function import *
from veni.module import Module


key = jax.random.PRNGKey(0)

def compare_params_shape(p, p_shape):
    if len(p) != len(p_shape):
        raise ValueError("Mismatched number of layers, check parameter generation")
    

    for (wp, bp), (ws,bs) in zip(p,p_shape):
        assert wp.shape == ws, f"Weights shape is not consistent with the one expected got {wp.shape}, expected {ws}"
        assert bp.shape == bs, f"Bias shape is not consistent with the one expected got {bp.shape}, expected {bs}"




    
class dummy_MLP(Module):
    def __init__(self,key):
        self.layers = Sequential(
            [   
                Linear(3,1,key),
                Softmax()
            ]
        )
    
        self.params = self.layers.generate_parameters()
    
    def forward(self, x, params):
        return self.layers(x,params)

def test_MLP_parameter_generation():
    m = dummy_MLP(key)
    p_shape = [ 
                ((3,1), (1,1)),
                ((0,) , (0,))   
            ]
    compare_params_shape(m.params, p_shape)

def test_mlp_fwd_pass():
    m = dummy_MLP(key)
    v = m(jnp.array([1.,2.,3.]), m.params)
    assert v == 1., f"Forward pass failed, got {v}, expected 1."

class dummy_CNN(Module):
    def __init__(self,key):
        self.layers = Sequential(
            [   
                Conv2D(1,1,3,1,'VALID',key),
                Softmax()
            ]
        )
    
        self.params = self.layers.generate_parameters()
    
    def forward(self, x, params):
        return self.layers(x,params)

def test_CNN_parameter_generation():
    m = dummy_CNN(key)
    p_shape = [ 
                ((1, 1, 3, 3), (1, 1, 1, 1)),
                ((0,), (0,)  )
            ]
    compare_params_shape(m.params, p_shape)

def test_CNN_fwd_pass():
    m = dummy_CNN(key)
    x = jnp.array([[float(i) for i in range(5)] for j in range(5)]).reshape(1,1,5,5)
    v = m(x,m.params)
    expected = jnp.array(
        [[[[0.03401766, 0.08532067, 0.21399502],
        [0.03401766, 0.08532067, 0.21399502],
        [0.03401766, 0.08532067, 0.21399502]]]])
    print(v)
    assert jnp.isclose(v,expected).all(), f"Forward pass CNN failed, got {v}, expected {expected}"


   