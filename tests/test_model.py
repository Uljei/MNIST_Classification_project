import numpy as np
from model import MLPModel

def test_model_initialization():
    model = MLPModel(10, 2, [5], 1, ['sigmoid', 'softmax'], "cross_entropy")
    assert len(model.layers) == 2
    assert model.layers[0].W.shape == (10, 5)
    assert model.layers[1].W.shape == (5, 2)

def test_model_forward():
    model = MLPModel(2, 3, [4], 1, ['sigmoid', 'softmax'], "cross_entropy")
    x = np.random.randn(10, 2)
    output = model.forward(x)
    assert output.shape == (10, 3)
    assert np.allclose(np.sum(output, axis=1), 1)

def test_model_predict():
    model = MLPModel(2, 3, [4], 1, ['sigmoid', 'softmax'], "cross_entropy")
    x = np.random.randn(10, 2)
    predictions = model.predict(x)
    assert predictions.shape == (10,)
    assert all(0 <= p <= 2 for p in predictions)

def test_model_evaluate():
    model = MLPModel(2, 3, [4], 1, ['sigmoid', 'softmax'], "cross_entropy")
    x = np.random.randn(10, 2)
    y = np.eye(3)[np.random.randint(0, 3, 10)]
    accuracy = model.evaluate(x, y)
    assert 0 <= accuracy <= 1