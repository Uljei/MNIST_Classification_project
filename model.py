import gzip
import pickle
import numpy as np
from Visualization import *
from Functions import *

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
            if self.num_layers == 0:  # no hidden layer
                self.layers.append(layer(self.input_size, self.output_size, self.activations[0]))
            else:
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
        if y.shape[1] == 4:
            y_pred = bit4_logits_to_int(y_hat)
            y_true = bit4_bits_to_int(y)
        else: 
            y_pred = np.argmax(y_hat, axis=1)
            y_true = np.argmax(y, axis=1)
        # else:
        #     raise ValueError("y must be one-hot (10) or 4-bit encoded (4)")
        return np.mean(y_pred == y_true)

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

# Experiment Runner
def run_experiment(
        label_mode='onehot', 
        layers = [784, 30, 10],
        activations = ['sigmoid', 'sigmoid'],
        loss = 'mse',
        epochs=15, lr=0.5, batch_size=32, seed=42, verbose=1
        ):
    # Ensure reproducibility
    np.random.seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist('mnist.pkl.gz', label_mode=label_mode)
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
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }

if __name__ == '__main__':
    set_seed(42)

    # Train both modes with quiet training output
    res_onehot = run_experiment() # adjust default params as needed
    with open('trained_mnist_model.pkl', 'wb') as f:
        pickle.dump(res_onehot['model'], f)
    print("Model saved: trained_mnist_model.pkl")

    res_bit4 = run_experiment('bit4', layers = [784, 32, 4], activations = ['sigmoid', None], loss = 'bce_logits')
    with open('trained_mnist_model_bit4.pkl', 'wb') as f:
        pickle.dump(res_bit4['model'], f)
    print("Model saved: trained_mnist_model_bit4.pkl")

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

    # FGSM comparisons
    eps = 0.3
    n_eval = 3000

    X_eval_10 = res_onehot['X_test'][:n_eval]
    Y_eval_10 = res_onehot['y_test'][:n_eval]
     # FGSM attack
    acc_clean_10, acc_adv_10, X_adv = plot_fgsm_comparison_grid(res_onehot['model'], X_eval_10, Y_eval_10, eps,
                                                         mode_title='One-hot', n=5,
                                                         save_path='fgsm_onehot.png')
    summarize_results('One-hot Attack Evaluation', acc_clean_10, acc_adv_10, eps)

    X_eval_b4 = res_bit4['X_test'][:n_eval]
    Y_eval_b4 = res_bit4['y_test'][:n_eval]
    acc_clean_b4, acc_adv_b4, _ = plot_fgsm_comparison_grid(
        res_bit4['model'], X_eval_b4, Y_eval_b4, eps,
        mode_title='Bit-4',
        decode_pred_fn=bit4_logits_to_int,
        decode_true_fn=bit4_bits_to_int,
        n=5,
        save_path='fgsm_bit4.png')
    summarize_results('Bit-4 Attack Evaluation', acc_clean_b4, acc_adv_b4, eps)

    # retrain one-hot model with FGSM adversarial examples
    loaded_model = res_onehot['model']
    X_mixed = np.vstack([res_onehot['X_train'], X_adv])
    y_mixed = np.vstack([res_onehot['y_train'], Y_eval_10])

    # shuffle data
    indices = np.random.permutation(X_mixed.shape[0])
    X_mixed = X_mixed[indices]
    y_mixed = y_mixed[indices]

    loaded_model.fit(X_mixed, y_mixed, res_onehot['X_val'], res_onehot['y_val'], epochs=30, lr=0.2, batch_size=10)
    test_acc = loaded_model.evaluate(res_onehot['X_test'], res_onehot['y_test'])
    
    print("\n=== Retrained Model with FGSM Adversarial Examples ===")
    print("Attack Model Final Test Accuracy:", test_acc)# Generate adversarial examples using FGSM
    with open('trained_mnist_attack_model.pkl', 'wb') as f:
        pickle.dump(loaded_model, f)
    print("Model have been saved: trained_mnist_model.pkl")

    # Plot Trining/Validation Loss Curve
    plot_loss_curve(loaded_model, save_path='loss_curve_retrained_model.png') 

    ## New Attack with DeepFool
    with open('trained_mnist_attack_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    eps_df = 5
    acc_clean_orig, acc_adv_orig = plot_df_grid(res_onehot['model'], X_eval_10, Y_eval_10, eps_df, n=5,
                                             save_path='df_original.png')
    summarize_results('Original Model DeepFool Attack', acc_clean_orig, acc_adv_orig, eps_df)

    acc_clean_df, acc_adv_df = plot_df_grid(loaded_model, X_eval_10, Y_eval_10, eps_df, n=5,
                                                         save_path='df_onehot.png')
    summarize_results('DeepFool Attack Evaluation', acc_clean_df, acc_adv_df, eps_df)
 
    print("\nArtifacts saved:")
    print(" - loss_curves_10out_vs_4bit.png")
    print(" - predictions_onehot.png")
    print(" - predictions_bit4.png")
    print(" - fgsm_onehot.png")
    print(" - fgsm_bit4.png")
    print(" - loss_curve_retrained_model.png")
