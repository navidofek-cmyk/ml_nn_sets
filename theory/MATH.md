# Matematické základy neuronových sítí

## 1. Co je neuronová síť?

Neuronová síť je funkce `f(x; θ)`, která mapuje vstup `x` na výstup `ŷ`.
Skládá se z **vrstev** — každá vrstva dělá jednoduchou lineární transformaci + nelineárnost.

```
x → [Vrstva 1] → [Vrstva 2] → ... → [Vrstva L] → ŷ
```

---

## 2. Jedna vrstva

Vstup: vektor `a` (výstup předchozí vrstvy nebo vstupní data)

```
z = W · a + b       ← lineární část (W = váhy, b = bias)
a_next = f(z)       ← nelineární část (aktivační funkce)
```

- `W` je matice tvaru `(výstupní_neurony × vstupní_neurony)`
- `b` je vektor tvaru `(výstupní_neurony)`
- `f` je aktivační funkce aplikovaná po prvcích

---

## 3. Aktivační funkce

### ReLU (Rectified Linear Unit)
```
f(z) = max(0, z)
f'(z) = 1 pokud z > 0, jinak 0
```
Použití: skryté vrstvy — jednoduchá, rychlá, funguje skvěle v praxi.

### Sigmoid
```
f(z) = 1 / (1 + e^(-z))
f'(z) = f(z) · (1 - f(z))
```
Použití: binární klasifikace (výstupní vrstva).

### Softmax (pro vícetřídní klasifikaci)
```
f(z_i) = e^(z_i) / Σ_j e^(z_j)
```
Převede vektor na pravděpodobnostní distribuci (součet = 1).

### Lineární (bez aktivace)
```
f(z) = z,   f'(z) = 1
```
Použití: výstupní vrstva pro regresi.

---

## 4. Loss funkce (ztrátová funkce)

Měří, jak moc se predikce `ŷ` liší od skutečnosti `y`.

### MSE (Mean Squared Error) — pro regresi
```
L = (1/n) · Σ (ŷ_i - y_i)²
∂L/∂ŷ_i = (2/n) · (ŷ_i - y_i)
```

### Cross-Entropy — pro klasifikaci
```
L = -Σ_i y_i · log(ŷ_i)
```
Se softmaxem se gradient zjednodušší na:
```
∂L/∂z_i = ŷ_i - y_i    (kde y je one-hot vektor)
```

---

## 5. Backpropagation (zpětné šíření)

Cíl: spočítat gradient loss vzhledem ke všem vahám (∂L/∂W, ∂L/∂b).
Použijeme **chain rule** (pravidlo derivování složené funkce).

### Výstupní vrstva L:
```
δ_L = ∂L/∂z_L = ∂L/∂a_L ⊙ f'(z_L)
```
(⊙ = Hadamardův součin, tj. element-wise násobení)

### Skrytá vrstva l:
```
δ_l = (W_{l+1}^T · δ_{l+1}) ⊙ f'(z_l)
```

### Gradienty vah:
```
∂L/∂W_l = δ_l · a_{l-1}^T     (vnější součin)
∂L/∂b_l = δ_l
```

---

## 6. SGD (Stochastic Gradient Descent)

Aktualizace vah po každém kroku:
```
W = W - α · ∂L/∂W
b = b - α · ∂L/∂b
```
kde `α` je **learning rate** (rychlost učení, typicky 0.001–0.1).

### Mini-batch SGD
Místo jednoho vzorku použijeme dávku (batch) velikosti `B`:
```
L_batch = (1/B) · Σ L(x_i, y_i)
```
Gradienty se zprůměrují přes dávku.

---

## 7. Inicializace vah

Váhy **nesmíme** inicializovat na nulu (síť by se nikdy nenaučila nic).

### Xavier/Glorot inicializace (pro sigmoid/tanh):
```
W ~ Uniform(-√(6/(fan_in + fan_out)), +√(6/(fan_in + fan_out)))
```

### He inicializace (pro ReLU):
```
W ~ Normal(0, √(2/fan_in))
```

---

## 8. Shrnutí trénovacího cyklu

```
Pro každou epochu:
  Pro každou dávku (batch):
    1. Forward pass:  ŷ = f(x; θ)
    2. Loss:          L = loss(ŷ, y)
    3. Backward pass: spočítej ∂L/∂θ (backprop)
    4. Update:        θ = θ - α · ∂L/∂θ
  Vypiš průměrnou loss pro epochu
```
