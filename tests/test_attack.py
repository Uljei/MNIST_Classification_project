import numpy as np
from model import MLPModel
from Functions import fgsm_attack, grad_input, deepfool_multiclass

def test_grad_input():
    model = MLPModel(5, 2, [3], 1, ['sigmoid', 'softmax'], "cross_entropy")
    x = np.random.rand(3, 5)
    y = np.eye(2)[np.random.randint(0, 2, 3)]
    gradient = grad_input(model, x, y)
    assert gradient.shape == x.shape

def test_fgsm_attack():
    model = MLPModel(5, 2, [3], 1, ['sigmoid', 'softmax'], "cross_entropy")
    x = np.random.rand(3, 5)
    y = np.eye(2)[np.random.randint(0, 2, 3)]
    x_adv = fgsm_attack(model, x, y, epsilon=0.1)
    assert x_adv.shape == x.shape
    assert np.all(x_adv >= 0) and np.all(x_adv <= 1)
    assert not np.allclose(x, x_adv)

def test_deepfool_multiclass_basic():
    
    model = MLPModel(5, 2, [3], 1, ['sigmoid', 'softmax'], "cross_entropy")
    x = np.random.rand(3, 5)
    y = np.eye(2)[np.random.randint(0, 2, 3)]
    x_adv = deepfool_multiclass(model, x, y, max_iter=10, epsilon=5.0)
    assert x_adv.shape == x.shape
    assert np.all(x_adv >= 0) and np.all(x_adv <= 1) 
    assert not np.allclose(x, x_adv)
