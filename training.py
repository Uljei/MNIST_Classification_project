import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 损失函数 (平方损失)
def mse_loss(y_true, y_pred):
    return np.mean(np.sum((y_true - y_pred) ** 2, axis=1)) / 2

# One-hot 编码
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

# 神经网络类
class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=30, output_size=10, lr=0.5):
        self.lr = lr
        # 初始化权重 (均匀分布)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)  # 输出层也用 Sigmoid
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        # 输出层误差
        dz2 = (self.a2 - y) * sigmoid_derivative(self.z2)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0, keepdims=True)

        # 隐藏层误差
        dz1 = (dz2 @ self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # 更新
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# 加载 MNIST (已下载 mnist.pkl.gz)
def load_mnist():
    with gzip.open("mnist.pkl.gz", "rb") as f:
        train_set, val_set, test_set = pickle.load(f, encoding="latin1")
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set
    return X_train, y_train, X_val, y_val, X_test, y_test

# 训练
def train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    y_train_oh = one_hot(y_train, 10)
    history = {"loss": [], "acc": [], "val_acc": []}
    for epoch in range(epochs):
        # 打乱数据
        perm = np.random.permutation(len(X_train))
        X_train, y_train_oh, y_train = X_train[perm], y_train_oh[perm], y_train[perm]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train_oh[i:i+batch_size]
            out = model.forward(X_batch)
            model.backward(X_batch, y_batch)

        # 计算准确率
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        acc = np.mean(train_pred == y_train)
        val_acc = np.mean(val_pred == y_val)
        loss = mse_loss(y_train_oh, model.forward(X_train))

        history["loss"].append(loss)
        history["acc"].append(acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, val_acc={val_acc:.4f}")
    return history

# 简单攻击 (FGSM)
def fgsm_attack(model, X, y_true, epsilon=0.1):
    y_true_oh = one_hot(y_true, 10)
    model.forward(X)
    # 输出层误差
    dz2 = (model.a2 - y_true_oh) * sigmoid_derivative(model.z2)
    grad_input = (dz2 @ model.W2.T) * sigmoid_derivative(model.z1) @ model.W1.T
    # 扰动
    X_adv = X + epsilon * np.sign(grad_input)
    return np.clip(X_adv, 0, 1)

# =================== 主程序 ===================
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    model = NeuralNetwork()
    history = train(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32)

    # 测试集准确率
    test_pred = model.predict(X_test)
    print("Test accuracy:", np.mean(test_pred == y_test))

    # 攻击实验
    X_adv = fgsm_attack(model, X_test[:100], y_test[:100], epsilon=0.2)
    adv_pred = model.predict(X_adv)
    print("攻击后准确率:", np.mean(adv_pred == y_test[:100]))