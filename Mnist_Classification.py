import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np


# Activation Func
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return (x > 0).astype(float)

# Loss Func
def mse(y, y_hat):
    return 0.5 * np.mean(np.sum((y - y_hat) ** 2, axis=1))

def mse_derivative(y, y_hat):
    return y_hat - y

def cross_entropy(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))

def cross_entropy_derivative(y, y_hat):
    return (y_hat - y) / y.shape[0]

# 4输出用了二元交叉熵
def bce_logits(y, logits):
    z = logits
    # numerically stable BCE with logits
    return np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))

def bce_logits_derivative(y, logits):
    # dL/dz = sigmoid(z) - y
    return sigmoid(logits) - y

# Network Layer
class layer():
    def __init__(self, n_input, n_output, activation=None):
        self.W = np.random.randn(n_input, n_output) / np.sqrt(n_input)
        self.b = np.zeros((1, n_output))
        self.activation = activation
        self.input = None
        self.output = None

    def forward(self, input):
        z = np.dot(input, self.W) + self.b
        if self.activation == 'sigmoid':
            self.output = sigmoid(z)
        elif self.activation == 'ReLU':
            self.output = ReLU(z)
        elif self.activation == 'tanh':
            self.output = tanh(z)
        elif self.activation == 'softmax':
            self.output = softmax(z)
        else:
            self.output = z
        self.input = input
        return self.output

    def backward(self, output_grad):
        if self.activation == 'sigmoid':
            output_grad = output_grad * sigmoid_derivative(self.output)
        elif self.activation == 'ReLU':
            output_grad = output_grad * ReLU_derivative(self.output)
        elif self.activation == 'tanh':
            output_grad = output_grad * tanh_derivative(self.output)
        self.W_grad = np.dot(self.input.T, output_grad)
        self.b_grad = np.sum(output_grad, axis=0, keepdims=True)
        return np.dot(output_grad, self.W.T)

# MLP Model
class MLPModel():
    def __init__(self, input_size, output_size, hidden_sizes, num_layers, activations, loss="mse"):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.activations = activations
        self.layers = []
        self.loss = loss
        self.train_losses = []  # training loss
        self.val_losses = []    # validation loss
        self.init_weights()

    def init_weights(self):
        self.layers.append(layer(self.input_size, self.hidden_sizes[0], self.activations[0]))
        for i in range(1, self.num_layers):
            self.layers.append(layer(self.hidden_sizes[i-1], self.hidden_sizes[i], self.activations[i]))
        self.layers.append(layer(self.hidden_sizes[-1], self.output_size, self.activations[-1]))

    def forward(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out

    def backward(self, y, y_hat):
        if self.loss == "mse":
            grad = mse_derivative(y, y_hat)
        elif self.loss == "cross_entropy":
            grad = cross_entropy_derivative(y, y_hat)
        elif self.loss == "bce_logits":
            grad = bce_logits_derivative(y, y_hat)
        else:
            raise ValueError("Unsupported loss function")
        for l in reversed(self.layers):
            grad = l.backward(grad)

    def update(self, lr):
        for l in self.layers:
            l.W -= lr * l.W_grad
            l.b -= lr * l.b_grad

    def fit(self, X, y, X_val, y_val, epochs, lr, batch_size, verbose=1):
        for epoch in range(epochs):
            idx = np.random.permutation(X.shape[0]) # stochastic gradient decent
            X, y = X[idx], y[idx]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_hat = self.forward(X_batch)
                self.backward(y_batch, y_hat)
                self.update(lr)

            # calculate training and validation loss every epoch
            y_hat_train = self.forward(X)
            y_hat_val = self.forward(X_val)
            if self.loss == "mse":
                train_loss = mse(y, y_hat_train)
                val_loss = mse(y_val, y_hat_val)
            elif self.loss == "cross_entropy":
                train_loss = cross_entropy(y, y_hat_train)  # y_hat_train is probs (softmax)
                val_loss = cross_entropy(y_val, y_hat_val)
            elif self.loss == "bce_logits":
                train_loss = bce_logits(y, y_hat_train)  # y_hat_train are logits
                val_loss = bce_logits(y_val, y_hat_val)
            else:
                raise ValueError("Unsupported loss")
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            train_acc = self.evaluate(X, y)
            val_acc = self.evaluate(X_val, y_val)
            if verbose:
                print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    def predict(self, X):
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)

    def evaluate(self, X, y):
        y_hat = self.forward(X)
        if y.shape[1] == 10:
            y_pred = np.argmax(y_hat, axis=1)
            y_true = np.argmax(y, axis=1)
        elif y.shape[1] == 4:
            y_pred = bit4_logits_to_int(y_hat)
            y_true = bit4_bits_to_int(y)
        else:
            raise ValueError("y must be one-hot (10) or 4-bit encoded (4)")
        return np.mean(y_pred == y_true)

# ---- 4输出支持 ----
def to_bit4(y_digits):
    y_digits = np.asarray(y_digits).astype(int)
    bits = ((y_digits[:, None] >> np.arange(4)) & 1)
    return bits.astype(float)

def bit4_logits_to_int(logits):
    """Decode 4-bit *logits* to digit int in [0,9]."""
    arr = np.asarray(logits)
    if arr.ndim == 1:
        arr = arr[None, :]
    probs = 1 / (1 + np.exp(-arr))
    bits = (probs >= 0.5).astype(int)
    vals = (bits * (2 ** np.arange(4))).sum(axis=1)
    return np.clip(vals, 0, 9)

def bit4_bits_to_int(bits):
    """Decode 4-bit *binary bits/probabilities* (targets) to digit int in [0,9].
    Accepts 0/1 or probabilities in [0,1]. Threshold at 0.5; **no sigmoid applied**.
    """
    arr = np.asarray(bits)
    if arr.ndim == 1:
        arr = arr[None, :]
    b = (arr >= 0.5).astype(int)
    vals = (b * (2 ** np.arange(4))).sum(axis=1)
    return np.clip(vals, 0, 9)

# Backward-compat alias
from_bit4 = bit4_logits_to_int

# Loading Data
def load_mnist(path='mnist.pkl.gz', label_mode='onehot'):
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f, encoding='latin1')
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set
    if label_mode == 'onehot':
        y_train = np.eye(10)[y_train]
        y_val = np.eye(10)[y_val]
        y_test = np.eye(10)[y_test]
    elif label_mode == 'bit4':
        y_train = to_bit4(y_train)
        y_val = to_bit4(y_val)
        y_test = to_bit4(y_test)
    else:
        raise ValueError("label_mode must be 'onehot' or 'bit4'")
    return X_train, y_train, X_val, y_val, X_test, y_test

# functions for attack
def grad_input(model, X, y):
    """
    returns dLoss/dX (same shape as X).
    For one-hot: y is one-hot; for 4-bit: y is 4-bit bits (0/1).
    """
    # forward
    y_hat = model.forward(X)

    # choose dL/d(output of last layer)
    if model.loss == "mse":
        grad = mse_derivative(y, y_hat)                # dL/dA_L  (A_L is activation output)
        last_is_logits = False
    elif model.loss == "cross_entropy":
        grad = cross_entropy_derivative(y, y_hat)      # dL/dA_L  (softmax+CE约化梯度)
        last_is_logits = False
    elif model.loss == "bce_logits":
        # y_hat are logits when activation=None on last layer
        grad = bce_logits_derivative(y, y_hat)         # dL/dZ_L  (Z_L is logits)
        last_is_logits = True
    else:
        raise ValueError("Unsupported loss")

    # backprop to input
    # 遍历各层（从后往前），最后一层若是logits，就不再乘激活导数
    first = True
    for l in reversed(model.layers):
        if first and last_is_logits:
            # do NOT multiply by activation' because output is logits already
            pass
        else:
            if l.activation == 'sigmoid':
                grad = grad * sigmoid_derivative(l.output)
            elif l.activation == 'ReLU':
                grad = grad * ReLU_derivative(l.output)
            elif l.activation == 'tanh':
                grad = grad * tanh_derivative(l.output)
        grad = np.dot(grad, l.W.T)
        first = False

    return grad

def fgsm_attack(model, X, y, epsilon=0.1):
    """
    FGSM: x_adv = x + ε·sign(∂L/∂x)
    """
    g = grad_input(model, X, y)
    x_adv = X + epsilon * np.sign(g)
    # limit to [0,1]
    x_adv = np.clip(x_adv, 0.0, 1.0)
    return x_adv

# 加了个一键跑（感觉有点乱所以这么写），不过感觉4输出也不需要这么多的样子
def run_experiment(label_mode='onehot', epochs=15, lr=0.3, batch_size=32, seed=42, verbose=1):
    # Ensure reproducibility
    np.random.seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist('mnist.pkl.gz', label_mode=label_mode)
    if label_mode == 'onehot':
        layers = [784, 30, 10]
        activations = ['sigmoid', 'softmax']
        loss = 'cross_entropy'
    else:
        layers = [784, 32, 4]
        # hidden uses sigmoid, output is linear (logits) for BCE-with-logits
        activations = ['sigmoid', None]
        loss = 'bce_logits'
    model = MLPModel(layers[0], layers[-1], layers[1:-1], len(layers)-2, activations, loss=loss)
    model.fit(X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose)
    test_acc = model.evaluate(X_test, y_test)
    return {
        'label_mode': label_mode,
        'test_acc': float(test_acc),
        'train_losses': model.train_losses,
        'val_losses': model.val_losses,
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
    }

# 还没写完，放了个思路周五写
def hyperparam_study(label_mode, configs):
    results = []
    for cfg in configs:
        epochs = cfg.get('epochs', 10)
        lr = cfg.get('lr', 0.3)
        batch_size = cfg.get('batch_size', 32)
        hidden = cfg.get('hidden', 30)
        # build layers and activations per label_mode
        if label_mode == 'onehot':
            layers = [784, hidden, 10]
            activations = ['sigmoid', 'softmax']
            loss = 'cross_entropy'
        else:
            layers = [784, hidden, 4]
            activations = ['sigmoid', None]
            loss = 'bce_logits'
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist('mnist.pkl.gz', label_mode=label_mode)
        model = MLPModel(layers[0], layers[-1], layers[1:-1], len(layers)-2, activations, loss=loss)
        model.fit(X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size)
        acc_val = model.evaluate(X_val, y_val)
        results.append({
            'label_mode': label_mode,
            'hidden': hidden,
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'val_acc': float(acc_val),
            'final_val_loss': float(model.val_losses[-1])
        })
        print(f"HP -> mode={label_mode}, hidden={hidden}, lr={lr}, bs={batch_size} | Val Acc={acc_val:.4f}")
    return results

#为了可复现用的，意义不大
def set_seed(seed=42):
    np.random.seed(seed)



def summarize_results(title, acc_clean, acc_adv, eps):
    print(f"\n=== {title} ===")
    print(f"Clean accuracy: {acc_clean:.4f}")
    print(f"FGSM accuracy (eps={eps}): {acc_adv:.4f}")


def plot_loss_curves_both(res_onehot, res_bit4, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(res_onehot['train_losses']) + 1), res_onehot['train_losses'], marker='o', label='Train (10-out)')
    plt.plot(range(1, len(res_onehot['val_losses']) + 1), res_onehot['val_losses'], marker='s', label='Val (10-out)')
    plt.plot(range(1, len(res_bit4['train_losses']) + 1), res_bit4['train_losses'], marker='^', label='Train (4-bit)')
    plt.plot(range(1, len(res_bit4['val_losses']) + 1), res_bit4['val_losses'], marker='v', label='Val (4-bit)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves: 10-output vs 4-bit')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_predictions_grid(model, X, y, mode_title, decode_pred_fn=None, decode_true_fn=None, n=5, save_path=None):
    idx = np.random.choice(X.shape[0], n, replace=False)
    Xs = X[idx]
    if decode_pred_fn is None and decode_true_fn is None:
        y_true = np.argmax(y[idx], axis=1)
        y_pred = model.predict(Xs)
    else:
        # use provided decoders
        y_true = decode_true_fn(y[idx])
        y_pred = decode_pred_fn(model.forward(Xs))
    fig, axes = plt.subplots(1, n, figsize=(2.2*n, 2.6))
    for i, ax in enumerate(axes):
        ax.imshow(Xs[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f"T:{int(y_true[i])} P:{int(y_pred[i])}")
    plt.suptitle(f"{mode_title}: Predictions (T=true, P=pred)")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_fgsm_comparison_grid(model, X, Y, eps, mode_title, decode_pred_fn=None, decode_true_fn=None, n=5, save_path=None):
    # compute clean accuracy
    if decode_pred_fn is None and decode_true_fn is None:
        logits_clean = model.forward(X)
        acc_clean = np.mean(np.argmax(logits_clean, 1) == np.argmax(Y, 1))
    else:
        y_pred_clean = decode_pred_fn(model.forward(X))
        y_true_clean = decode_true_fn(Y)
        acc_clean = np.mean(y_pred_clean == y_true_clean)

    # attack
    X_adv = fgsm_attack(model, X, Y, epsilon=eps)

    # compute adv accuracy
    if decode_pred_fn is None and decode_true_fn is None:
        logits_adv = model.forward(X_adv)
        acc_adv = np.mean(np.argmax(logits_adv, 1) == np.argmax(Y, 1))
    else:
        y_pred_adv = decode_pred_fn(model.forward(X_adv))
        acc_adv = np.mean(y_pred_adv == y_true_clean)

    # visualize pairs
    idx = np.random.choice(X.shape[0], n, replace=False)
    plt.figure(figsize=(2.2*n, 4.4))
    for i, j in enumerate(idx):
        plt.subplot(2, n, i + 1)
        plt.imshow(X[j].reshape(28, 28), cmap='gray')
        plt.title('Clean')
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(X_adv[j].reshape(28, 28), cmap='gray')
        plt.title('FGSM')
        plt.axis('off')
    plt.suptitle(f"{mode_title} FGSM (eps={eps})")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.show()

    return acc_clean, acc_adv

if __name__ == '__main__':
    set_seed(42)

    # Train both modes with quiet training output
    res_onehot = run_experiment('onehot', epochs=15, lr=0.3, batch_size=32, seed=42, verbose=0)
    res_bit4   = run_experiment('bit4',   epochs=15, lr=0.3, batch_size=32, seed=42, verbose=0)

    # Print final test accuracies
    print("\n=== One-hot and Bit-4: Final Test Accuracy ===")
    print(f"One-hot (10 outputs) Test Acc: {res_onehot['test_acc']:.4f}")
    print(f"Bit-4  (4 outputs)  Test Acc: {res_bit4['test_acc']:.4f}")

    # Loss curve comparison (saved)
    plot_loss_curves_both(res_onehot, res_bit4, save_path='loss_curves_10out_vs_4bit.png')

    # Prediction grids (saved)
    plot_predictions_grid(res_onehot['model'], res_onehot['X_test'], res_onehot['y_test'],
                          mode_title='One-hot model', n=5,
                          save_path='predictions_onehot.png')

    plot_predictions_grid(res_bit4['model'], res_bit4['X_test'], res_bit4['y_test'],
                          mode_title='Bit-4 model',
                          decode_pred_fn=bit4_logits_to_int,
                          decode_true_fn=bit4_bits_to_int,
                          n=5,
                          save_path='predictions_bit4.png')

    # FGSM comparisons (saved)
    eps = 0.2
    n_eval = 1000

    X_eval_10 = res_onehot['X_test'][:n_eval]
    Y_eval_10 = res_onehot['y_test'][:n_eval]
    acc_clean_10, acc_adv_10 = plot_fgsm_comparison_grid(res_onehot['model'], X_eval_10, Y_eval_10, eps,
                                                         mode_title='One-hot', n=5,
                                                         save_path='fgsm_onehot.png')
    summarize_results('One-hot Attack Evaluation', acc_clean_10, acc_adv_10, eps)

    X_eval_b4 = res_bit4['X_test'][:n_eval]
    Y_eval_b4 = res_bit4['y_test'][:n_eval]
    acc_clean_b4, acc_adv_b4 = plot_fgsm_comparison_grid(
        res_bit4['model'], X_eval_b4, Y_eval_b4, eps,
        mode_title='Bit-4',
        decode_pred_fn=bit4_logits_to_int,
        decode_true_fn=bit4_bits_to_int,
        n=5,
        save_path='fgsm_bit4.png')
    summarize_results('Bit-4 Attack Evaluation', acc_clean_b4, acc_adv_b4, eps)

    print("\nArtifacts saved:")
    print(" - loss_curves_10out_vs_4bit.png")
    print(" - predictions_onehot.png")
    print(" - predictions_bit4.png")
    print(" - fgsm_onehot.png")
    print(" - fgsm_bit4.png")