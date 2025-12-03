import numpy as np
import gzip
import random

# MNIST BETÖLTŐ

def load_mnist_local():
    def load_images(path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28*28).astype(np.float32)
        data = data / 255.0   # normalizálás
        return data

    def load_labels(path):
        with gzip.open(path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    x_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")

    x_test = load_images("t10k-images-idx3-ubyte.gz")
    y_test = load_labels("t10k-labels-idx1-ubyte.gz")

    return (x_train, y_train), (x_test, y_test)

# Aktivációk és deriváltak

def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    return dA * (Z > 0)

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# Dense Layer
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation="relu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        limit = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * limit
        self.b = np.zeros((1, output_dim))

        self.Z = None
        self.A_prev = None
        self.A = None

    def forward(self, A_prev):
        self.A_prev = A_prev                        # (m, input_dim)
        self.Z = A_prev @ self.W + self.b           # (m, output_dim)

        if self.activation == "relu":
            self.A = relu(self.Z)
        elif self.activation == "sigmoid":
            self.A = sigmoid(self.Z)
        elif self.activation == "softmax":
            self.A = softmax(self.Z)
        else:
            self.A = self.Z  # linear

        return self.A

    def backward(self, dA, learning_rate):
        m = self.A_prev.shape[0]

        if self.activation == "relu":
            dZ = relu_backward(dA, self.Z)
        elif self.activation == "sigmoid":
            dZ = sigmoid_backward(dA, self.Z)
        else:
            dZ = dA

        dW = self.A_prev.T @ dZ / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ self.W.T

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_prev

    def backward_from_dZ(self, dZ, learning_rate):
        m = self.A_prev.shape[0]

        dW = self.A_prev.T @ dZ / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ self.W.T

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_prev

# MLP NETWORK
class MLP:
    def __init__(self, layer_sizes, activations):
        assert len(layer_sizes) - 1 == len(activations)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i])
            )

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A  # utolsó réteg softmax-ja

    @staticmethod
    def compute_loss(y_onehot, y_pred):
        m = y_onehot.shape[0]
        eps = 1e-12
        loss = -np.sum(y_onehot * np.log(y_pred + eps)) / m
        return loss

    def fit_batch(self, X_batch, y_batch, lr):
        m = X_batch.shape[0]
        num_classes = self.layers[-1].output_dim

        # One-hot
        y_onehot = np.zeros((m, num_classes))
        y_onehot[np.arange(m), y_batch] = 1

        # Forward
        y_pred = self.forward(X_batch)

        # Loss
        loss = self.compute_loss(y_onehot, y_pred)

        # Backward: softmax + cross entropy -> dZ = y_pred - y_onehot
        dZ = y_pred - y_onehot

        # Output layer
        dA_prev = self.layers[-1].backward_from_dZ(dZ, lr)

        # Hidden layers
        for layer in reversed(self.layers[:-1]):
            dA_prev = layer.backward(dA_prev, lr)

        return loss

    def fit(self, X, y, epochs=5, batch_size=64, lr=0.01, verbose=True):
        m = X.shape[0]
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(m)
            X = X[idx]
            y = y[idx]

            epoch_loss = 0
            num_batches = 0

            for start in range(0, m, batch_size):
                end = start + batch_size
                if end > m:
                    end = m

                Xb = X[start:end]
                yb = y[start:end]

                batch_loss = self.fit_batch(Xb, yb, lr)
                epoch_loss += batch_loss
                num_batches += 1

            epoch_loss /= num_batches
            if verbose:
                print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

# FUTTATÁS

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_local()

    mlp = MLP(
        layer_sizes=[784, 128, 64, 10],
        activations=["relu", "relu", "softmax"]
    )

    mlp.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        lr=0.01
    )

    acc = mlp.accuracy(x_test, y_test)
    print(f"\nTeszt pontosság: {acc*100:.2f}%")

    # Példa egy mintára
    idx = random.randrange(0,len(x_test) - 1)
    pred = mlp.predict(x_test[idx:idx+1])[0]
    print(f"Valódi szám: {y_test[idx]}, Predikció: {pred}")
