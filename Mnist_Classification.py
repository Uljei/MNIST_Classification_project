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
def hyperparam_study(
    label_mode,
    configs,
    seed=42,
    early_stopping=True,
    patience=5,
    eval_metric='val_acc',      # 也可用 'val_loss' 作为选择标准
    return_best_model=False,    # True 时返回 { 'best_model': model, ... }
    fgsm_eval=False,            # True 时在验证集上做一次 FGSM 稳健性评估
    fgsm_eps_list=(0.2,),       # 可传多个 eps
    n_eval_fgsm=1000            # FGSM 评估子集大小
):
    """
    网格搜索 / 批量实验器：
      - 支持 hidden 为 int 或 list（多隐藏层）
      - 支持 activations 为 list，长度 = 隐藏层数 + 1（最后一层激活）
        * 若未提供 activations，则按 label_mode 使用默认：
          onehot:   ['sigmoid', 'softmax']
          bit4:     ['sigmoid', None]  (BCE-with-logits)
      - 早停：基于 eval_metric（val_acc 或 val_loss），patience 达到则停止
      - 可选：在验证集做 FGSM 稳健性评估（记录 clean 与 adv accuracy）
    返回：按 eval_metric 最优优先排序的结果列表（每个元素是 dict）
          若 return_best_model=True，会额外带上 'best_model'
    """
    rng = np.random.RandomState(seed)
    all_results = []
    best_pack = None

    # 统一载数据（每个 cfg 可单独设 seed 则重复载入也 OK）
    def _load(label_mode):
        return load_mnist('mnist.pkl.gz', label_mode=label_mode)

    for cfg in configs:
        # 解析配置
        _seed      = cfg.get('seed', seed)
        rng.seed(_seed)
        np.random.seed(_seed)

        epochs     = int(cfg.get('epochs', 15))
        lr         = float(cfg.get('lr', 0.3))
        batch_size = int(cfg.get('batch_size', 32))

        hidden     = cfg.get('hidden', 30)
        if isinstance(hidden, int):
            hidden_sizes = [hidden]
        else:
            hidden_sizes = list(hidden)

        # 设定激活（允许覆盖）
        acts = cfg.get('activations', None)
        if acts is None:
            if label_mode == 'onehot':
                # 隐藏层全部用 sigmoid，输出 softmax
                acts = ['sigmoid'] * len(hidden_sizes) + ['softmax']
                loss = 'cross_entropy'
                out_size = 10
            else:
                # 隐藏层 sigmoid，输出 logits（None）+ BCE-with-logits
                acts = ['sigmoid'] * len(hidden_sizes) + [None]
                loss = 'bce_logits'
                out_size = 4
        else:
            # 用户自定义激活，推断输出大小 & loss
            acts = list(acts)
            if label_mode == 'onehot':
                out_size = 10
                loss = 'cross_entropy' if acts[-1] == 'softmax' else cfg.get('loss', 'cross_entropy')
            else:
                out_size = 4
                loss = 'bce_logits' if acts[-1] is None else cfg.get('loss', 'bce_logits')

            if len(acts) != len(hidden_sizes) + 1:
                raise ValueError("activations 长度需等于 隐藏层数 + 1（包含输出层激活）。")

        # 构图
        layers = [784] + hidden_sizes + [out_size]
        num_layers = len(layers) - 2  # 隐藏层数

        # 加载数据
        X_train, y_train, X_val, y_val, X_test, y_test = _load(label_mode)

        # 初始化模型
        model = MLPModel(
            input_size=layers[0],
            output_size=layers[-1],
            hidden_sizes=layers[1:-1],
            num_layers=num_layers,
            activations=acts,
            loss=loss
        )

        # ----- 训练循环（带早停） -----
        best_val_acc  = -np.inf
        best_val_loss = np.inf
        best_epoch    = -1
        no_imp        = 0

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # 打乱
            idx = rng.permutation(X_train.shape[0])
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            # 小批量 SGD
            for i in range(0, X_shuf.shape[0], batch_size):
                Xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]
                y_hat = model.forward(Xb)
                model.backward(yb, y_hat)
                model.update(lr)

            # 计算当轮的 train/val loss
            y_hat_tr  = model.forward(X_train)
            y_hat_val = model.forward(X_val)

            if loss == "mse":
                trL = mse(y_train, y_hat_tr);  vaL = mse(y_val, y_hat_val)
            elif loss == "cross_entropy":
                trL = cross_entropy(y_train, y_hat_tr);  vaL = cross_entropy(y_val, y_hat_val)
            elif loss == "bce_logits":
                trL = bce_logits(y_train, y_hat_tr);     vaL = bce_logits(y_val, y_hat_val)
            else:
                raise ValueError("Unsupported loss.")

            train_losses.append(trL);  val_losses.append(vaL)

            # 计算当轮的 val_acc
            val_acc = model.evaluate(X_val, y_val)

            # 依据指标判定是否更新“最优”
            improved = False
            if eval_metric == 'val_acc':
                if val_acc > best_val_acc:
                    best_val_acc, best_val_loss = val_acc, vaL
                    best_epoch, improved = epoch, True
            elif eval_metric == 'val_loss':
                if vaL < best_val_loss:
                    best_val_loss, best_val_acc = vaL, val_acc
                    best_epoch, improved = epoch, True
            else:
                raise ValueError("eval_metric 仅支持 'val_acc' 或 'val_loss'")

            # 早停计数
            if improved:
                no_imp = 0
                # 可选：如需“真正回滚”参数，可在此深拷贝 model（当前实现记录指标即可）
                best_snapshot = {
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_loss': vaL,
                }
            else:
                no_imp += 1
                if early_stopping and no_imp >= patience:
                    # print(f"[EarlyStop] epoch={epoch+1} no_improve={no_imp}")
                    break

        # 选一个统一的“测试集表现”做记录（注意：未回滚参数，则是最后一轮的）
        test_acc = model.evaluate(X_test, y_test)

        # 可选：验证集 FGSM 稳健性评估
        fgsm_records = []
        if fgsm_eval:
            n_eval = min(n_eval_fgsm, X_val.shape[0])
            Xe = X_val[:n_eval]; Ye = y_val[:n_eval]
            if label_mode == 'onehot':
                clean_acc = np.mean(np.argmax(model.forward(Xe), 1) == np.argmax(Ye, 1))
                for eps in fgsm_eps_list:
                    Xadv = fgsm_attack(model, Xe, Ye, epsilon=eps)
                    adv_acc = np.mean(np.argmax(model.forward(Xadv), 1) == np.argmax(Ye, 1))
                    fgsm_records.append({'eps': float(eps), 'clean_acc': float(clean_acc), 'adv_acc': float(adv_acc)})
            else:
                y_pred_clean = bit4_logits_to_int(model.forward(Xe))
                y_true_clean = bit4_bits_to_int(Ye)
                clean_acc = float(np.mean(y_pred_clean == y_true_clean))
                for eps in fgsm_eps_list:
                    Xadv = fgsm_attack(model, Xe, Ye, epsilon=eps)
                    y_pred_adv = bit4_logits_to_int(model.forward(Xadv))
                    adv_acc = float(np.mean(y_pred_adv == y_true_clean))
                    fgsm_records.append({'eps': float(eps), 'clean_acc': clean_acc, 'adv_acc': adv_acc})

        result = {
            'label_mode': label_mode,
            'hidden': hidden if isinstance(hidden, int) else tuple(hidden),
            'activations': tuple(acts),
            'loss': loss,
            'lr': float(lr),
            'batch_size': int(batch_size),
            'epochs_run': best_epoch + 1 if early_stopping else epochs,
            'final_val_loss': float(val_losses[-1]),
            'final_val_acc': float(model.evaluate(X_val, y_val)),
            'best_val_loss': float(best_val_loss),
            'best_val_acc': float(best_val_acc),
            'test_acc': float(test_acc),
            'seed': int(_seed),
        }
        if fgsm_eval:
            result['fgsm_val'] = fgsm_records

        pack = {'result': result, 'model': model}
        all_results.append(result)

        # 维护全局最优
        better = False
        if best_pack is None:
            better = True
        else:
            if eval_metric == 'val_acc':
                better = (result['best_val_acc'] > best_pack['result']['best_val_acc'])
            else:
                better = (result['best_val_loss'] < best_pack['result']['best_val_loss'])
        if better:
            best_pack = pack

    # 排序输出
    if eval_metric == 'val_acc':
        all_results.sort(key=lambda d: d['best_val_acc'], reverse=True)
    else:
        all_results.sort(key=lambda d: d['best_val_loss'])

    if return_best_model:
        return {
            'results': all_results,
            'best_model': best_pack['model'],
            'best_result': best_pack['result']
        }
    return all_results


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

    cfgs = [
        # 10 类 one-hot：softmax + CE
        {'hidden': 30, 'epochs': 30, 'lr': 0.3, 'batch_size': 32},
        {'hidden': [128, 64], 'epochs': 30, 'lr': 0.2, 'batch_size': 64, 'activations': ['ReLU', 'ReLU', 'softmax']},
        {'hidden': [64, 64, 32], 'epochs': 40, 'lr': 0.15, 'batch_size': 64,
         'activations': ['tanh', 'tanh', 'tanh', 'softmax']},

        # 4-bit：logits(None) + BCE-with-logits
        {'hidden': 32, 'epochs': 30, 'lr': 0.3, 'batch_size': 32},
        {'hidden': [64, 32], 'epochs': 35, 'lr': 0.25, 'batch_size': 64, 'activations': ['sigmoid', 'sigmoid', None]},
        {'hidden': [128, 64, 32], 'epochs': 40, 'lr': 0.2, 'batch_size': 64,
         'activations': ['ReLU', 'ReLU', 'ReLU', None]},
    ]

    res10 = hyperparam_study('onehot', cfgs[:3], early_stopping=True, patience=6,
                             eval_metric='val_acc', fgsm_eval=True, fgsm_eps_list=(0.1, 0.2), return_best_model=True)
    resb4 = hyperparam_study('bit4', cfgs[3:], early_stopping=True, patience=6,
                             eval_metric='val_acc', fgsm_eval=True, fgsm_eps_list=(0.1, 0.2), return_best_model=True)

    print("Top (onehot):", res10['best_result'])
    print("Top (bit4):  ", resb4['best_result'])
