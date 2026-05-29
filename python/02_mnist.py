"""
Část 2: Klasifikace MNIST
Python ekvivalent k cpp/02_mnist/main.cpp

PyTorch automaticky stáhne MNIST data přes torchvision.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np

print("=== Část 2: MNIST klasifikace ===\n")

# ── Data ──────────────────────────────────────────────────────────────────────
# torchvision stáhne MNIST automaticky
transform = T.Compose([
    T.ToTensor(),                        # PIL → tensor (1×28×28), škála 0–1
    T.Normalize((0.1307,), (0.3081,)),   # standardizace (μ, σ celého datasetu)
])

train_ds = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_ds  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=256, shuffle=False)

print(f"Trénovací vzorky: {len(train_ds)}")
print(f"Testovací vzorky: {len(test_ds)}\n")

# ── Síť ───────────────────────────────────────────────────────────────────────
# Identická architektura jako C++ verze: 784 → 256 → 128 → 10
# nn.Flatten: přemění (B, 1, 28, 28) → (B, 784)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),              # bez softmax — CrossEntropyLoss ji aplikuje interně
)

print("Architektura:")
total = sum(p.numel() for p in model.parameters())
print(model)
print(f"Celkem parametrů: {total:,}\n")

# ── Trénink ───────────────────────────────────────────────────────────────────
# CrossEntropyLoss = Softmax + NLLLoss (numericky stabilní)
# Na vstupu očekává logity (ne softmax!) a celočíselné štítky
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn   = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

print(f"{'Epocha':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8}")
print("-------|------------|-----------|----------")

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        logits = model(X)
        loss   = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_acc = evaluate(model, train_loader)
    test_acc  = evaluate(model, test_loader)
    avg_loss  = total_loss / len(train_loader)
    print(f"{epoch:6d} | {avg_loss:10.4f} | {train_acc*100:8.2f}% | {test_acc*100:7.2f}%")

print(f"\nFinální testovací přesnost: {evaluate(model, test_loader)*100:.2f}%")
print("(Benchmark: ~97-98% s jednoduchou MLP)\n")

# ── Ukázka predikcí ───────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

model.eval()
X_sample, y_sample = next(iter(test_loader))
with torch.no_grad():
    preds = model(X_sample[:16]).argmax(dim=1)

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i, ax in enumerate(axes.flat):
    img = X_sample[i].squeeze().numpy()
    ax.imshow(img, cmap="gray")
    color = "green" if preds[i] == y_sample[i] else "red"
    ax.set_title(f"pred={preds[i]}", color=color, fontsize=9)
    ax.axis("off")
plt.suptitle("MNIST predikce (zelená=správně, červená=špatně)")
plt.tight_layout()
plt.savefig("02_mnist.png")
print("Ukázka uložena jako 02_mnist.png")
