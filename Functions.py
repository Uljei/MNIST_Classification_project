import numpy as np
def set_seed(seed=42):
    np.random.seed(seed)
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

# functions for 4 output BCE with logits
def bce_logits_derivative(y, logits):
    # dL/dz = sigmoid(z) - y
    return sigmoid(logits) - y

def to_bit4(y_digits):
    '''Convert digit labels in [0,9] to 4-bit binary representation.'''
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

 # functions for attack
def grad_input(model, X, y):
    """
    Given model, input X and output y (one-hot encoded).,
    returns dLoss/dX (same shape as X).
    """
    # forward
    y_hat = model.forward(X)

    # derivative of the loss with respect to the output
    bce_mark = False
    if model.loss == "mse":
        grad = mse_derivative(y, y_hat)
    elif model.loss == "cross_entropy":
        grad = cross_entropy_derivative(y, y_hat)
    elif model.loss == "bce_logits":
        bce_mark = True
        last_layer = model.layers[-1]
        z_last = np.dot(model.layers[-2].output, last_layer.W) + last_layer.b
        grad = bce_logits_derivative(y, z_last)  # use logits
    else:
        raise ValueError("Unsupported loss")
    
    # Propagate the gradient from the last layer back to the input
    First = True
    for l in reversed(model.layers):
        if First and bce_mark:
            pass
        else:
            # First multiply by the derivative of the activation
            if l.activation == 'sigmoid':
                grad = grad * sigmoid_derivative(l.output)
            elif l.activation == 'ReLU':
                grad = grad * ReLU_derivative(l.output)
            elif l.activation == 'tanh':
                grad = grad * tanh_derivative(l.output)
        # For softmax we skip extra handling here
        First = False
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

def deepfool_multiclass(model, X, y, max_iter=10, epsilon=0.3):
    """
    One-hot encoded DeepFool attack
    """
    X_adv = X.copy()
    
    for i in range(X.shape[0]):
        x = X[i:i+1]
        y_true = y[i:i+1]
        true_label = np.argmax(y_true)  
        
        r = np.zeros_like(x)
        
        for iter in range(max_iter):
            logits = model.forward(x + r)
            grad_orig = grad_input(model, x + r, y_true)
            # check if the classification has changed
            pred_label = np.argmax(logits)
            if pred_label != true_label:
                break
            pert = np.inf
            w_best = None
            num_classes = model.output_size
            for k in range(num_classes):
                if k == true_label:
                    continue
                grad_k = grad_input(model, x + r, np.eye(num_classes)[[k]])
                w_k = grad_k - grad_orig
                f_k = logits[0, true_label] - logits[0, k]

                pert_k = f_k / (np.linalg.norm(w_k) + 1e-8)

                if pert_k < pert:
                    pert = pert_k
                    w_best = w_k

            # if no valid boundary is found, stop
            if w_best is None or np.linalg.norm(w_best) < 1e-8:
                r += epsilon * np.sign(np.random.randn(*x.shape)) * 0.1
                break

            r_i = (pert / (np.linalg.norm(w_best) + 1e-8)) * w_best
            r += r_i

            if np.linalg.norm(r) > epsilon:
                r = r / np.linalg.norm(r) * epsilon
                break

        X_adv[i] = np.clip(x + r, 0.0, 1.0)
    
    X_adv = np.clip(X_adv, 0.0, 1.0)
    return X_adv