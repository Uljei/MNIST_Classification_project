import numpy as np
from Visualization import plot_loss_curves_both, summarize_results, plot_predictions_grid, plot_loss_curve

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

class DummyModel:
    """a simple model for testing"""
    def __init__(self):
        self.train_losses = [1.0, 0.8, 0.6]
        self.val_losses = [1.1, 0.9, 0.7]

    def forward(self, X):
        return np.random.randn(X.shape[0], 10)

    def predict(self, X):
        return np.random.randint(0, 10, size=X.shape[0])

def test_plot_loss_curve_runs(tmp_path):
    model = DummyModel()
    save_file = tmp_path / "loss.png"
    # functionnally check
    plot_loss_curve(model, save_path=save_file, title_prefix="Test")
    assert save_file.exists()

def test_plot_predictions_grid_runs(tmp_path):
    model = DummyModel()
    X = np.random.rand(10, 28*28)
    y = np.eye(10)[np.random.choice(10, 10)]
    save_file = tmp_path / "pred.png"
    plot_predictions_grid(model, X, y, mode_title="Test", save_path=save_file)
    assert save_file.exists()
