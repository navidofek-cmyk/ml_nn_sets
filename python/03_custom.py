"""
Část 3: Vlastní dataset
Python ekvivalent k cpp/03_custom/main.cpp

Ukázáno:
  - Načtení vlastních dat (spirála jako příklad, snadno nahradíš CSV)
  - Normalizace (StandardScaler)
  - Train/val/test split
  - Early stopping
  - Binární metriky (precision, recall, F1)
  - Vizualizace rozhodovací hranice
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print("=== Část 3: Vlastní dataset (two-moons) ===\n")

# ── Generování dat — two-moons (nahraď svým CSV) ─────────────────────────────
# make_moons: dvě půlkružnice (lineárně neseparovatelné, ale snadno naučitelné)
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
print(f"Dataset: {X.shape[0]} vzorků, {X.shape[1]} features, 2 třídy")
print(f"Třída 0: {(y==0).sum()}, Třída 1: {(y==1).sum()}\n")

# ── Jak načíst vlastní CSV ────────────────────────────────────────────────────
# import pandas as pd
# df = pd.read_csv("tvůj_soubor.csv")
# X = df.drop("target", axis=1).values
# y = df["target"].values

# ── Split: 64% train / 16% val / 20% test ────────────────────────────────────
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")

# ── Normalizace (fit POUZE na train!) ─────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Tensory
def to_tensor(x, y):
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).reshape(-1, 1))

Xt, yt   = to_tensor(X_train, y_train)
Xv, yv   = to_tensor(X_val,   y_val)
Xte, yte = to_tensor(X_test,  y_test)

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xt, yt), batch_size=32, shuffle=True)

# ── Síť ───────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(2, 64),  nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),  nn.Sigmoid(),  # výstup ∈ (0,1) = pravděpodobnost třídy 1
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.BCELoss()

# ── Early stopping ────────────────────────────────────────────────────────────
best_val_loss = float("inf")
patience, no_improve = 20, 0

print(f"{'Epocha':>6} | {'Train BCE':>9} | {'Val BCE':>9} | {'Val Acc':>7}")
print("-------|-----------|-----------|--------")

for epoch in range(1, 501):
    model.train()
    total = 0
    for Xb, yb in loader:
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    model.eval()
    with torch.no_grad():
        val_pred = model(Xv)
        val_loss = loss_fn(val_pred, yv).item()
        val_acc  = ((val_pred > 0.5).float() == yv).float().mean().item()

    if epoch % 50 == 0 or epoch == 1:
        print(f"{epoch:6d} | {total/len(loader):9.4f} | {val_loss:9.4f} | {val_acc*100:6.2f}%")

    # Early stopping
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), "/tmp/best_model.pt")  # uložíme nejlepší model
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping v epoše {epoch}")
            break

# Načti nejlepší model
model.load_state_dict(torch.load("/tmp/best_model.pt"))

# ── Finální evaluace ──────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_pred = (model(Xte) > 0.5).long().numpy().flatten()
y_true = y_test.astype(int)

print("\n=== Výsledky na testovací sadě ===")
print(classification_report(y_true, test_pred, target_names=["Třída 0", "Třída 1"]))

# ── Vizualizace rozhodovací hranice ──────────────────────────────────────────
model.eval()
h = 0.05
x1 = np.arange(-2.5, 2.5, h)
x2 = np.arange(-2.5, 2.5, h)
xx1, xx2 = np.meshgrid(x1, x2)
grid = scaler.transform(np.c_[xx1.ravel(), xx2.ravel()])
with torch.no_grad():
    Z = model(torch.tensor(grid, dtype=torch.float32)).numpy().reshape(xx1.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, Z, levels=50, cmap="RdBu", alpha=0.7)
plt.scatter(X[y==0, 0], X[y==0, 1], s=10, c="blue", alpha=0.5, label="Třída 0")
plt.scatter(X[y==1, 0], X[y==1, 1], s=10, c="red",  alpha=0.5, label="Třída 1")
plt.colorbar(label="P(třída 1)")
plt.legend(); plt.title("Rozhodovací hranice NN (spirála)")
plt.tight_layout()
plt.savefig("03_custom.png")
print("Rozhodovací hranice uložena jako 03_custom.png")
