from model import run_experiment

def test_run_experiment_onehot():
    '''test onehot experiment'''
    res = run_experiment(
        label_mode='onehot',
        layers=[784, 10], 
        activations=['softmax'],
        loss='cross_entropy',
        epochs=1,
        verbose=0
    )
    assert 'test_acc' in res
    assert 0 <= res['test_acc'] <= 1

def test_run_experiment_bit4():
    '''test bit4 experiment'''
    res = run_experiment(
        label_mode='bit4',
        layers=[784, 4],
        activations=[None],
        loss='bce_logits',
        epochs=1,
        verbose=0
    )
    assert 'test_acc' in res
    assert 0 <= res['test_acc'] <= 1