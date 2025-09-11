import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

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
    return (y_hat - y)

def cross_entropy(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))

def cross_entropy_derivative(y, y_hat):
    return (y_hat - y) / y.shape[0]

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
        else:
            raise ValueError("Unsupported loss function")
        for l in reversed(self.layers):
            grad = l.backward(grad)

    def update(self, lr):
        for l in self.layers:
            l.W -= lr * l.W_grad
            l.b -= lr * l.b_grad

    def fit(self, X, y, X_val, y_val, epochs, lr, batch_size):
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
            else:
                train_loss = cross_entropy(y, y_hat_train)
                val_loss = cross_entropy(y_val, y_hat_val)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            train_acc = self.evaluate(X, y)
            val_acc = self.evaluate(X_val, y_val)
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
            y_pred = from_bit4(y_hat)
            y_true = from_bit4(y)
        else:
            raise ValueError("y must be one-hot (10) or 4-bit encoded (4)")
        return np.mean(y_pred == y_true)

# ---- 4-bit output support ----
def to_bit4(y_digits):
    y_digits = np.asarray(y_digits).astype(int)
    bits = ((y_digits[:, None] >> np.arange(4)) & 1)
    return bits.astype(float)

def from_bit4(logits_or_probs):
    arr = np.asarray(logits_or_probs)
    if arr.ndim == 1:
        arr = arr[None, :]
    probs = 1 / (1 + np.exp(-arr))
    bits = (probs >= 0.5).astype(int)
    vals = (bits * (2 ** np.arange(4))).sum(axis=1)
    return np.clip(vals, 0, 9)

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
    Given model, input X and output y (one-hot encoded).，
    returns dLoss/dX (same shape as X).
    """
    # forward
    y_hat = model.forward(X)

    # derivative of the loss with respect to the output
    if model.loss == "mse":
        grad = mse_derivative(y, y_hat)
    elif model.loss == "cross_entropy":
        grad = cross_entropy_derivative(y, y_hat)
    else:
        raise ValueError("Unsupported loss")

    # Propagate the gradient from the last layer back to the input
    for l in reversed(model.layers):
        # First multiply by the derivative of the activation
        if l.activation == 'sigmoid':
            grad = grad * sigmoid_derivative(l.output)
        elif l.activation == 'ReLU':
            grad = grad * ReLU_derivative(l.output)
        elif l.activation == 'tanh':
            grad = grad * tanh_derivative(l.output)
        # For softmax we skip extra handling here
        grad = np.dot(grad, l.W.T)

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

def run_experiment(label_mode='onehot', epochs=15, lr=0.3, batch_size=32):
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist('mnist.pkl.gz', label_mode=label_mode)
    if label_mode == 'onehot':
        layers = [784, 30, 10]
        activations = ['sigmoid', 'softmax']
        loss = 'cross_entropy'
    else:
        layers = [784, 32, 4]
        activations = ['sigmoid', 'sigmoid']
        loss = 'mse'
    model = MLPModel(layers[0], layers[-1], layers[1:-1], len(layers)-2, activations, loss=loss)
    model.fit(X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, batch_size=batch_size)
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
            activations = ['sigmoid', 'sigmoid']
            loss = 'mse'
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


if __name__ == '__main__':
    # ---- Task 2: Compare 10-output vs 4-output ----
    res_onehot = run_experiment('onehot', epochs=15, lr=0.3, batch_size=32)
    res_bit4 = run_experiment('bit4', epochs=15, lr=0.3, batch_size=32)
    print("\n=== One-hot and Bit-4 Comparison ===")
    print(f"One-hot (10 outputs) Test Acc: {res_onehot['test_acc']:.4f}")
    print(f"Bit-4  (4 outputs)  Test Acc: {res_bit4['test_acc']:.4f}")

    # ---- Show Test Dataset Prediction Results (one-hot model) ----
    indices = np.random.choice(res_onehot['X_test'].shape[0], 5, replace=False)
    X_samples = res_onehot['X_test'][indices]
    y_true = np.argmax(res_onehot['y_test'][indices], axis=1)
    y_pred = res_onehot['model'].predict(X_samples)

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X_samples[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
        ax.set_title(f"T:{y_true[i]} P:{y_pred[i]}")
    plt.suptitle("One-hot model: Test samples (T=true, P=pred)")
    plt.show()

    # ---- FGSM clean vs adversarial (one-hot model) ----
    print("-" * 20, "Attack Evaluation (one-hot)", "-" * 20)
    n_eval = 1000
    X_eval = res_onehot['X_test'][:n_eval]
    Y_eval = res_onehot['y_test'][:n_eval]

    logits_clean = res_onehot['model'].forward(X_eval)
    acc_clean = np.mean(np.argmax(logits_clean, 1) == np.argmax(Y_eval, 1))
    print(f"Accuracy on clean {n_eval} samples: {acc_clean:.4f}")

    eps = 0.2  # you can adjust 0.05/0.1/0.2/0.3
    X_adv = fgsm_attack(res_onehot['model'], X_eval, Y_eval, epsilon=eps)
    logits_adv = res_onehot['model'].forward(X_adv)
    acc_adv = np.mean(np.argmax(logits_adv, 1) == np.argmax(Y_eval, 1))
    print(f"Accuracy under FGSM attack (eps={eps}): {acc_adv:.4f}")

    idx = np.random.choice(n_eval, 5, replace=False)
    plt.figure(figsize=(10, 4))
    for i, j in enumerate(idx):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_eval[j].reshape(28, 28), cmap="gray")
        plt.title("Clean")
        plt.axis("off")
        plt.subplot(2, 5, i + 6)
        plt.imshow(X_adv[j].reshape(28, 28), cmap="gray")
        plt.title("FGSM")
        plt.axis("off")
    plt.suptitle(f"FGSM eps={eps}")
    plt.show()

    # ---- Show Test Dataset Prediction Results (bit-4 model) ----
    indices = np.random.choice(res_bit4['X_test'].shape[0], 5, replace=False)
    X_samples = res_bit4['X_test'][indices]
    y_true_b4 = from_bit4(res_bit4['y_test'][indices])
    y_pred_b4 = from_bit4(res_bit4['model'].forward(X_samples))

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X_samples[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
        ax.set_title(f"T:{int(y_true_b4[i])} P:{int(y_pred_b4[i])}")
    plt.suptitle("Bit-4 model: Test samples (T=true, P=pred)")
    plt.show()

    # ---- FGSM clean vs adversarial (bit-4 model) ----
    print("-" * 20, "Attack Evaluation (bit-4)", "-" * 20)
    n_eval = 1000
    X_eval_b4 = res_bit4['X_test'][:n_eval]
    Y_eval_b4 = res_bit4['y_test'][:n_eval]

    # Accuracy on clean samples
    y_pred_clean = from_bit4(res_bit4['model'].forward(X_eval_b4))
    y_true_clean = from_bit4(Y_eval_b4)
    acc_clean_b4 = np.mean(y_pred_clean == y_true_clean)
    print(f"Accuracy on clean {n_eval} samples: {acc_clean_b4:.4f}")

    # FGSM attack
    eps = 0.2
    X_adv_b4 = fgsm_attack(res_bit4['model'], X_eval_b4, Y_eval_b4, epsilon=eps)
    y_pred_adv = from_bit4(res_bit4['model'].forward(X_adv_b4))
    acc_adv_b4 = np.mean(y_pred_adv == y_true_clean)
    print(f"Accuracy under FGSM attack (eps={eps}): {acc_adv_b4:.4f}")

    # Visualize 5 pairs
    idx = np.random.choice(n_eval, 5, replace=False)
    plt.figure(figsize=(10, 4))
    for i, j in enumerate(idx):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_eval_b4[j].reshape(28, 28), cmap="gray")
        plt.title("Clean")
        plt.axis("off")
        plt.subplot(2, 5, i + 6)
        plt.imshow(X_adv_b4[j].reshape(28, 28), cmap="gray")
        plt.title("FGSM")
        plt.axis("off")
    plt.suptitle(f"Bit-4 FGSM eps={eps}")
    plt.show()

    # Plot example loss curves for the second run
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(res_bit4['train_losses']) + 1), res_bit4['train_losses'], marker='o', label='Train (bit4)')
    plt.plot(range(1, len(res_bit4['val_losses']) + 1), res_bit4['val_losses'], marker='s', label='Val (bit4)')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.title('Loss Curve (bit4)');
    plt.legend();
    plt.grid(True);
    plt.show()