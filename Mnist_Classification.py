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
        y_pred = np.argmax(y_hat, axis=1)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)

# Loading Data
def load_mnist(path='mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f, encoding='latin1')
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set
    y_train = np.eye(10)[y_train]
    y_val = np.eye(10)[y_val]
    y_test = np.eye(10)[y_test]
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    # choose label mode from
    LABEL_MODE = 'onehot'
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist('mnist.pkl.gz')

    layers = [784, 30, 10]
    activations = ['sigmoid', 'sigmoid']
    model = MLPModel(layers[0], layers[-1], layers[1:-1], len(layers)-2, activations, loss="mse")  # Use other loss func

    model.fit(X_train, y_train, X_val, y_val, epochs=30, lr=0.5, batch_size=10)

    test_acc = model.evaluate(X_test, y_test)
    print("Final Test Accuracy:", test_acc)

    # Plot Trining/Validation Loss Curve
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(model.train_losses)+1), model.train_losses, marker='o', label="Train Loss")
    plt.plot(range(1, len(model.val_losses)+1), model.val_losses, marker='s', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model.loss.upper()} Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Show Test Dataset Prediction Results
    for image in range(5):
        indices = np.random.choice(X_test.shape[0], 5, replace=False)
        X_samples = X_test[indices]
        y_true = np.argmax(y_test[indices], axis=1)
        y_pred = model.predict(X_samples)

        fig, axes = plt.subplots(1, 5, figsize=(12,3))
        for i, ax in enumerate(axes):
            ax.imshow(X_samples[i].reshape(28,28), cmap="gray")
            ax.axis("off")
            ax.set_title(f"T:{y_true[i]} P:{y_pred[i]}")
        plt.show()
