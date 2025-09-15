import numpy as np
from Functions import *
def test_sigmoid():
    x = np.array([-1, 0, 1])
    y = sigmoid(x)
    expected = np.array([0.26894142, 0.5, 0.73105858])
    assert np.allclose(y, expected)

def test_sigmoid_derivative():
    x = np.array([0.2, 0.5, 0.8])
    deriv = sigmoid_derivative(x)
    expected = x * (1 - x)
    assert np.allclose(deriv, expected)

def test_tanh():
    x = np.array([-1, 0, 1])
    y = tanh(x)
    expected = np.array([-0.76159416, 0, 0.76159416])
    assert np.allclose(y, expected)

def test_tanh_derivative():
    x = np.array([-0.5, 0, 0.5])
    deriv = tanh_derivative(x)
    expected = 1 - np.tanh(x)**2
    assert np.allclose(deriv, expected)

def test_relu():
    x = np.array([-1, 0, 1])
    y = ReLU(x)
    expected = np.array([0, 0, 1])
    assert np.allclose(y, expected)

def test_relu_derivative():
    x = np.array([-1, 0, 1])
    deriv = ReLU_derivative(x)
    expected = np.array([0, 0, 1])
    assert np.allclose(deriv, expected)

def test_softmax():
    x = np.array([[1, 2, 3]])
    y = softmax(x)
    assert np.allclose(np.sum(y, axis=1), 1)
    expected = np.array([[0.09003057, 0.24472847, 0.66524096]])
    assert np.allclose(y, expected)
def test_mse():
    y = np.array([[0, 1], [1, 0]])
    y_hat = np.array([[0.2, 0.8], [0.9, 0.1]])
    loss = mse(y, y_hat)
    expected = 0.5 * np.mean(np.sum((y - y_hat) ** 2, axis=1))
    assert np.isclose(loss, expected)

def test_mse_derivative():
    y = np.array([[0, 1], [1, 0]])
    y_hat = np.array([[0.2, 0.8], [0.9, 0.1]])
    deriv = mse_derivative(y, y_hat)
    expected = y_hat - y
    assert np.allclose(deriv, expected)

def test_cross_entropy():
    y = np.array([[0, 1], [1, 0]])
    y_hat = np.array([[0.2, 0.8], [0.9, 0.1]])
    loss = cross_entropy(y, y_hat)
    assert loss > 0

def test_cross_entropy_derivative():
    y = np.array([[0, 1], [1, 0]])
    y_hat = np.array([[0.2, 0.8], [0.9, 0.1]])
    deriv = cross_entropy_derivative(y, y_hat)
    expected = (y_hat - y) / y.shape[0]
    assert np.allclose(deriv, expected)

def test_bce_logits():
    y = np.array([[0, 1], [1, 0]])
    logits = np.array([[0.5, -0.5], [0.1, 0.9]])
    loss = bce_logits(y, logits)
    assert loss > 0

def test_bce_logits_derivative():
    y = np.array([[0, 1], [1, 0]])
    logits = np.array([[0.5, -0.5], [0.1, 0.9]])
    deriv = bce_logits_derivative(y, logits)
    expected = sigmoid(logits) - y
    assert np.allclose(deriv, expected)

def test_to_bit4_and_back():
    digits = np.array([0, 1, 5, 9])
    bits = to_bit4(digits)
    assert bits.shape == (4, 4)
    recovered = bit4_bits_to_int(bits)
    assert np.all(recovered == digits)


def test_bit4_logits_to_int():
    logits = np.array([[10, -10, -10, -10]])  
    val = bit4_logits_to_int(logits)
    assert val[0] == 1

    logits = np.array([[-10, 10, -10, -10]]) 
    val = bit4_logits_to_int(logits)
    assert val[0] == 2

def test_bit4_bits_to_int_probabilities():
    probs = np.array([[0.6, 0.7, 0.2, 0.1]]) 
    val = bit4_bits_to_int(probs)
    assert val[0] == 3