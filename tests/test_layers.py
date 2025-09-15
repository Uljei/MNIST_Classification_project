import numpy as np
from model import layer

def test_layer_initialization():
    l = layer(10, 5, 'sigmoid')
    assert l.W.shape == (10, 5)
    assert l.b.shape == (1, 5)
    assert l.activation == 'sigmoid'

def test_layer_forward_sigmoid():
    l = layer(2, 3, 'sigmoid')
    x = np.random.randn(5, 2)
    output = l.forward(x)
    assert output.shape == (5, 3)
    assert np.all(output >= 0) and np.all(output <= 1)

def test_layer_forward_relu():
    l = layer(2, 3, 'ReLU')
    x = np.random.randn(5, 2)
    output = l.forward(x)
    assert output.shape == (5, 3)
    assert np.all(output >= 0)

def test_layer_forward_softmax():
    l = layer(2, 3, 'softmax')
    x = np.random.randn(5, 2)
    output = l.forward(x)
    assert output.shape == (5, 3)
    assert np.allclose(np.sum(output, axis=1), 1)

def test_layer_backward():
    l = layer(2, 3, 'sigmoid')
    x = np.random.randn(5, 2)
    l.forward(x)
    output_grad = np.random.randn(5, 3)
    input_grad = l.backward(output_grad)
    assert input_grad.shape == x.shape
    assert l.W_grad.shape == l.W.shape
    assert l.b_grad.shape == l.b.shape