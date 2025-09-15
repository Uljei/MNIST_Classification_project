import matplotlib.pyplot as plt
import numpy as np
from Visualization import plot_loss_curves_both, summarize_results, plot_df_grid
from model import MLPModel

def test_summarize_results(capsys):
    summarize_results('Test', 0.9, 0.1, 0.2)
    captured = capsys.readouterr()
    assert "Test" in captured.out
    assert "0.9000" in captured.out
    assert "0.1000" in captured.out

def test_plot_loss_curves_both():
    """ Basic functionality test for plot_loss_curves_both."""
    res_onehot = {'train_losses': [0.5, 0.4], 'val_losses': [0.6, 0.5]}
    res_bit4 = {'train_losses': [0.5, 0.4], 'val_losses': [0.6, 0.5]}
    try:
        plot_loss_curves_both(res_onehot, res_bit4)
        assert True
    except Exception as e:
        assert False, f"plot_loss_curves_both failed: {e}"
