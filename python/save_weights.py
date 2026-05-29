"""Natrénuje všechny PyTorch modely a uloží váhy do weights/"""
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from pathlib import Path

OUT = Path(__file__).parent.parent / "weights"
OUT.mkdir(exist_ok=True)

# ── 1. Regrese ────────────────────────────────────────────────
print("=== 1. Regrese ===")
rng = np.random.default_rng(42)
x = rng.uniform(-1, 1, 1000)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 1000)
Xt = torch.tensor(x.reshape(-1,1), dtype=torch.float32)
Yt = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

model_reg = nn.Sequential(
    nn.Linear(1,32), nn.ReLU(),
    nn.Linear(32,32), nn.ReLU(),
    nn.Linear(32,1)
)
opt = torch.optim.SGD(model_reg.parameters(), lr=0.01)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xt, Yt), batch_size=64, shuffle=True)

for epoch in range(500):
    for xb, yb in loader:
        loss = nn.MSELoss()(model_reg(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()

torch.save(model_reg.state_dict(), OUT / "regression_pt.pth")
print(f"  Uloženo: weights/regression_pt.pth  (test MSE: {nn.MSELoss()(model_reg(Xt), Yt).item():.4f})")

# ── 2. Two-moons ──────────────────────────────────────────────
print("=== 2. Two-moons ===")
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
scaler = StandardScaler().fit(X[:800])
Xn = scaler.transform(X)
Xt2 = torch.tensor(Xn, dtype=torch.float32)
Yt2 = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

model_moon = nn.Sequential(
    nn.Linear(2,64), nn.ReLU(),
    nn.Linear(64,64), nn.ReLU(),
    nn.Linear(64,1), nn.Sigmoid()
)
opt2 = torch.optim.Adam(model_moon.parameters(), lr=0.001)
loader2 = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xt2, Yt2), batch_size=64, shuffle=True)

for epoch in range(300):
    for xb, yb in loader2:
        loss = nn.BCELoss()(model_moon(xb), yb)
        opt2.zero_grad(); loss.backward(); opt2.step()

acc = ((model_moon(Xt2) > 0.5).float() == Yt2).float().mean().item()
torch.save(model_moon.state_dict(), OUT / "moons_pt.pth")
# Ulož i scaler parametry (potřebné pro inferenci)
np.save(OUT / "moons_scaler_mean.npy", scaler.mean_)
np.save(OUT / "moons_scaler_std.npy",  scaler.scale_)
print(f"  Uloženo: weights/moons_pt.pth  (acc: {acc*100:.1f}%)")

# ── 3. MNIST ──────────────────────────────────────────────────
print("=== 3. MNIST ===")
transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
train_ds = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_ds  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=256, shuffle=False)

model_mnist = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256), nn.ReLU(),
    nn.Linear(256,128), nn.ReLU(),
    nn.Linear(128,10)
)
opt3 = torch.optim.SGD(model_mnist.parameters(), lr=0.1, momentum=0.9)

for epoch in range(20):
    model_mnist.train()
    for X, y in train_loader:
        loss = nn.CrossEntropyLoss()(model_mnist(X), y)
        opt3.zero_grad(); loss.backward(); opt3.step()

model_mnist.eval()
correct = sum((model_mnist(X).argmax(1)==y).sum().item() for X,y in test_loader)
acc_mnist = correct / len(test_ds)
torch.save(model_mnist.state_dict(), OUT / "mnist_pt.pth")
print(f"  Uloženo: weights/mnist_pt.pth  (acc: {acc_mnist*100:.2f}%)")

print("\n✓ Hotovo — všechny váhy v weights/")
