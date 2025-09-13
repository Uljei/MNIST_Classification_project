import numpy as np
import pytest
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
    
    model = MLPModel(5, 3, [4], 1, ['sigmoid', 'softmax'], "cross_entropy")
    
    x = np.random.rand(2, 5)
    y = np.eye(3)[[0, 1]]
    
    x_adv = deepfool_multiclass(model, x, y, max_iter=10, epsilon=2.0)
   
    assert x_adv.shape == x.shape
    assert np.all(x_adv >= 0) and np.all(x_adv <= 1)
    
    assert not np.allclose(x, x_adv)

def test_deepfool_multiclass_effectiveness():
   
    model = MLPModel(5, 3, [8], 1, ['sigmoid', 'softmax'], "cross_entropy")
    
    x_train = np.random.rand(100, 5)
    y_train = np.eye(3)[np.random.randint(0, 3, 100)]
    
    for _ in range(30):
        y_pred = model.forward(x_train)
        model.backward(y_train, y_pred)
        for l in model.layers:
            l.W -= 0.1 * l.W_grad
            l.b -= 0.1 * l.b_grad

    x_test = np.random.rand(3, 5)
    y_test = np.eye(3)[[0, 1, 2]]
    
    x_adv = deepfool_multiclass(model, x_test, y_test, max_iter=30, epsilon=2.0)
    
    assert x_adv.shape == x_test.shape
    assert np.all(x_adv >= 0) and np.all(x_adv <= 1)
    
    perturbation_norm = np.linalg.norm(x_adv - x_test)
    assert perturbation_norm > 1e-6, f"No perturbation generated: {perturbation_norm}"

def test_deepfool_multiclass_epsilon_limit():
  
    model = MLPModel(5, 3, [4], 1, ['sigmoid', 'softmax'], "cross_entropy")
    
    x = np.random.rand(2, 5)
    y = np.eye(3)[[0, 1]]  
    
    epsilon = 0.1
    x_adv = deepfool_multiclass(model, x, y, max_iter=10, epsilon=epsilon)
    
    perturbation = np.linalg.norm(x_adv - x, ord=2, axis=1)
    assert np.all(perturbation <= epsilon + 1e-6), f"Perturbation {perturbation} exceeds epsilon {epsilon}"

def test_deepfool_multiclass_with_bit4():
    
    model = MLPModel(5, 4, [4], 1, ['sigmoid', None], "bce_logits")
    
    from Functions import to_bit4
    y_digits = np.array([3, 7]) 
    y = to_bit4(y_digits)
    
    x = np.random.rand(2, 5)
    
    x_adv = deepfool_multiclass(model, x, y, max_iter=5, epsilon=0.5)
    
    assert x_adv.shape == x.shape
    assert np.all(x_adv >= 0) and np.all(x_adv <= 1)