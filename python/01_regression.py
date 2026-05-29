"""
Část 1: Regrese — sin funkce
Python ekvivalent k cpp/01_regression/main.cpp

Ukázány jsou DVĚ verze:
  A) NumPy — od nuly, aby bylo vidět jak to funguje
  B) PyTorch — jak to vypadá v praxi
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Generování dat ────────────────────────────────────────────────────────────
def generate_data(N=1000, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, N)
    y = np.sin(2 * np.pi * x) + rng.normal(0, noise, N)
    return x.reshape(1, N), y.reshape(1, N)  # (1×N)

X_train, Y_train = generate_data(1000, seed=42)
X_test,  Y_test  = generate_data(200,  seed=99)

print("=== Část 1: Regrese ===\n")
print(f"Trénovací vzorky: {X_train.shape[1]}")
print(f"Testovací vzorky: {X_test.shape[1]}\n")


# ══════════════════════════════════════════════════════════════════════════════
# VERZE A: NumPy (ručně implementovaný backprop)
# Stejná logika jako C++ kód
# ══════════════════════════════════════════════════════════════════════════════
print("─── Verze A: NumPy (od nuly) ───\n")

def relu(z):       return np.maximum(0, z)
def relu_d(z):     return (z > 0).astype(float)
def mse(yh, y):    return np.mean((yh - y)**2)
def mse_grad(yh, y): return 2 * (yh - y) / yh.size

class NumpyNet:
    def __init__(self, layer_sizes, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            # He init
            w = rng.normal(0, np.sqrt(2/fan_in), (fan_out, fan_in))
            self.W.append(w)
            self.b.append(np.zeros((fan_out, 1)))
        self.z_cache = []
        self.a_cache = []

    def forward(self, x):
        self.z_cache, self.a_cache = [], [x]
        a = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ a + b
            self.z_cache.append(z)
            # Poslední vrstva: lineární; ostatní: ReLU
            a = z if i == len(self.W)-1 else relu(z)
            self.a_cache.append(a)
        return a

    def backward(self, grad):
        for l in range(len(self.W)-1, -1, -1):
            if l < len(self.W)-1:
                grad = grad * relu_d(self.z_cache[l])
            dW = grad @ self.a_cache[l].T
            db = grad.sum(axis=1, keepdims=True) / grad.shape[1]
            if l > 0:
                grad = self.W[l].T @ grad
            self.W[l] -= self.lr * dW
            self.b[l] -= self.lr * db

# Trénink
net_np = NumpyNet([1, 32, 32, 1], lr=0.01)
for epoch in range(1, 501):
    # Mini-batch SGD
    B = 64
    idx = np.random.permutation(X_train.shape[1])
    total_loss = 0
    for start in range(0, X_train.shape[1], B):
        batch = idx[start:start+B]
        Xb, Yb = X_train[:, batch], Y_train[:, batch]
        yh = net_np.forward(Xb)
        total_loss += mse(yh, Yb)
        net_np.backward(mse_grad(yh, Yb))

    if epoch % 100 == 0:
        yh_test = net_np.forward(X_test)
        print(f"Epocha {epoch:4d} | Train MSE: {total_loss/(X_train.shape[1]//B):.6f} "
              f"| Test MSE: {mse(yh_test, Y_test):.6f}")

print(f"\nFinální test MSE (NumPy): {mse(net_np.forward(X_test), Y_test):.6f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# VERZE B: PyTorch
# Stejná úloha, ale PyTorch se stará o backprop automaticky
# ══════════════════════════════════════════════════════════════════════════════
print("─── Verze B: PyTorch ───\n")

# Tensor z numpy
Xt  = torch.tensor(X_train.T, dtype=torch.float32)  # (N, 1)
Yt  = torch.tensor(Y_train.T, dtype=torch.float32)
Xvt = torch.tensor(X_test.T,  dtype=torch.float32)
Yvt = torch.tensor(Y_test.T,  dtype=torch.float32)

# Definice sítě — naprosto ekvivalentní C++ verzi
model = nn.Sequential(
    nn.Linear(1, 32),   nn.ReLU(),
    nn.Linear(32, 32),  nn.ReLU(),
    nn.Linear(32, 1),              # bez aktivace = lineární výstup
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn   = nn.MSELoss()

dataset = torch.utils.data.TensorDataset(Xt, Yt)
loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(1, 501):
    model.train()
    total = 0
    for Xb, Yb in loader:
        pred = model(Xb)
        loss = loss_fn(pred, Yb)
        optimizer.zero_grad()
        loss.backward()       # PyTorch spočítá backprop automaticky
        optimizer.step()
        total += loss.item()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(Xvt), Yvt).item()
        print(f"Epocha {epoch:4d} | Train MSE: {total/len(loader):.6f} "
              f"| Test MSE: {test_loss:.6f}")

# Vizualizace
model.eval()
with torch.no_grad():
    x_plot = torch.linspace(-1, 1, 300).reshape(-1, 1)
    y_plot = model(x_plot).numpy()

plt.figure(figsize=(10, 4))
plt.scatter(X_test[0], Y_test[0], s=10, alpha=0.5, label="Data")
plt.plot(x_plot.numpy(), y_plot, 'r-', lw=2, label="Predikce sítě")
plt.plot(x_plot.numpy(), np.sin(2*np.pi*x_plot.numpy()), 'g--', lw=1, label="sin(2πx)")
plt.legend(); plt.title("Regrese: sin(2πx)"); plt.tight_layout()
plt.savefig("01_regression.png"); print("\nGraf uložen jako 01_regression.png")
