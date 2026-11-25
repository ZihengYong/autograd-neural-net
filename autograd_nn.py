"""
Reverse-Mode Automatic Differentiation Engine for Neural Networks
=================================================================
A lightweight, pure-NumPy reverse-mode automatic differentiation (autograd)
engine supporting dynamic computation graphs, topological backpropagation,
broadcasting gradient reduction, and finite-difference gradient verification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


class Tensor:
    """2D tensor node supporting reverse-mode automatic differentiation."""

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @staticmethod
    def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Reduce broadcasted gradient axes back to target_shape."""
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for axis, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (other * -1.0)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def bce_loss(self, targets):
        """Numerically stable binary cross-entropy loss from logits."""
        t = targets.data if isinstance(targets, Tensor) else np.asarray(targets, dtype=np.float64)
        z = self.data
        loss_val = np.mean(np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z))))
        out = Tensor(loss_val, (self,), "bce")

        def _backward():
            if self.requires_grad:
                sig = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
                self.grad += ((sig - t) / z.size) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Reverse-mode topological traversal applying the chain rule."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


class MLP:
    """Multi-Layer Perceptron built directly on custom Tensor nodes."""

    def __init__(self, layer_dims, seed=42):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:]):
            scale = np.sqrt(2.0 / (d_in + d_out))
            self.weights.append(Tensor(rng.normal(0.0, scale, size=(d_in, d_out))))
            self.biases.append(Tensor(np.zeros((1, d_out))))

    def parameters(self):
        return self.weights + self.biases

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0.0)

    def forward(self, x):
        h = x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = h.tanh()
        return h


def finite_difference_check(seed=7, eps=1e-6):
    """Compare analytical AD gradients against central finite differences."""
    rng = np.random.default_rng(seed)
    X_chk = rng.normal(size=(16, 2))
    y_chk = rng.integers(0, 2, size=(16, 1)).astype(np.float64)

    model = MLP([2, 8, 6, 1], seed=seed)
    loss = model.forward(X_chk).bce_loss(y_chk)
    model.zero_grad()
    loss.backward()

    max_abs_err = 0.0
    max_rel_err = 0.0
    for param in model.parameters():
        ad_grad = param.grad.copy()
        num_grad = np.zeros_like(param.data)
        it = np.nditer(param.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = param.data[idx]
            param.data[idx] = orig + eps
            l_plus = model.forward(X_chk).bce_loss(y_chk).data
            param.data[idx] = orig - eps
            l_minus = model.forward(X_chk).bce_loss(y_chk).data
            param.data[idx] = orig
            num_grad[idx] = (l_plus - l_minus) / (2.0 * eps)
            it.iternext()

        abs_err = np.max(np.abs(ad_grad - num_grad))
        rel_err = np.linalg.norm(ad_grad - num_grad) / (
            np.linalg.norm(ad_grad) + np.linalg.norm(num_grad) + 1e-15
        )
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)

    return max_abs_err, max_rel_err


def make_two_moons(n_samples=320, noise=0.18, seed=42):
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2
    t1 = np.linspace(0, np.pi, n_half)
    t2 = np.linspace(0, np.pi, n_samples - n_half)
    x1 = np.column_stack([np.cos(t1), np.sin(t1)])
    x2 = np.column_stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([x1, x2]) + rng.normal(0.0, noise, size=(n_samples, 2))
    y = np.vstack([np.zeros((n_half, 1)), np.ones((n_samples - n_half, 1))])
    return X, y


def main():
    # 1. Central finite-difference gradient verification
    max_abs, max_rel = finite_difference_check()
    print(f"[Gradient Check] max_abs_err = {max_abs:.3e}, max_rel_err = {max_rel:.3e}")

    # 2. Train 3-layer MLP [2 -> 24 -> 24 -> 1] on non-linear Two-Moons dataset
    X_train, y_train = make_two_moons(n_samples=320, noise=0.18, seed=42)
    mlp = MLP([2, 24, 24, 1], seed=42)
    lr = 0.25
    epochs = 800
    losses = []

    for _ in range(epochs):
        loss = mlp.forward(X_train).bce_loss(y_train)
        losses.append(float(loss.data))
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= lr * p.grad

    preds = (1.0 / (1.0 + np.exp(-mlp.forward(X_train).data)) >= 0.5).astype(float)
    acc = np.mean(preds == y_train)
    print(f"[MLP Training] Initial BCE = {losses[0]:.4f}, Final BCE = {losses[-1]:.4f}, Train Acc = {acc*100:.2f}%")

    # 3. Plot training curve and learned decision boundary
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)
    axes[0].plot(losses, color="#1f77b4", linewidth=2.0)
    axes[0].set_title("Training Loss (Reverse-Mode AD + SGD)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1.0, 1.5, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs = (1.0 / (1.0 + np.exp(-mlp.forward(grid).data))).reshape(xx.shape)
    axes[1].contourf(xx, yy, probs, levels=25, cmap="RdBu_r", alpha=0.75)
    axes[1].contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5)
    axes[1].scatter(X_train[y_train.ravel() == 0, 0], X_train[y_train.ravel() == 0, 1], c="#2b5c8f", edgecolors="w", s=28, label="Class 0")
    axes[1].scatter(X_train[y_train.ravel() == 1, 0], X_train[y_train.ravel() == 1, 1], c="#b8332a", edgecolors="w", s=28, label="Class 1")
    axes[1].set_title(f"Learned Decision Boundary (Acc = {acc*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/ad_training_benchmark.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
