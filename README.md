# Neural Networks od nuly — C++ + Python (uv)

[![Teorie – web](https://img.shields.io/badge/Teorie-web-6366f1)](https://navidofek-cmyk.github.io/ml_nn_sets)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navidofek-cmyk/ml_nn_sets/blob/main/colab_tutorial.ipynb)

Kompletní tutorial: od matematiky přes implementaci v C++ až po PyTorch ekvivalenty.

## Struktura

```
nn_sets/
├── theory/MATH.md          # Matematické základy
├── cpp/
│   ├── common/             # Sdílené hlavičky (matrix, aktivace, síť)
│   ├── 01_regression/      # Část 1: Regrese
│   ├── 02_mnist/           # Část 2: Klasifikace (MNIST)
│   └── 03_custom/          # Část 3: Vlastní dataset
└── python/                 # Python ekvivalenty (uv + PyTorch)
```

## Pořadí čtení

1. `theory/MATH.md` — matematika (volitelné, ale doporučené)
2. `cpp/common/` — základní stavební kameny
3. `cpp/01_regression/` — první síť
4. `cpp/02_mnist/` — klasifikace
5. `cpp/03_custom/` — vlastní data
6. `python/` — totéž v PyTorch

## Jak spustit C++

```bash
# Každá část má vlastní Makefile
cd cpp/01_regression && make && ./nn_regression
cd cpp/02_mnist     && make && ./nn_mnist      # nejdřív stáhni data (viz níže)
cd cpp/03_custom    && make && ./nn_custom
```

## Jak stáhnout MNIST

```bash
cd cpp/02_mnist && bash download_mnist.sh
```

## Jak spustit Python (uv)

```bash
cd python
uv sync                  # nainstaluje závislosti
uv run python 01_regression.py
uv run python 02_mnist.py
uv run python 03_custom.py
```

## Požadavky C++

- C++17 nebo novější (`g++ -std=c++17`)
- žádné externí knihovny — čistá STL

## Co se naučíš

| Část | Téma | Klíčové koncepty |
|------|------|-----------------|
| 1    | Regrese | forward pass, MSE loss, backprop, SGD |
| 2    | MNIST klasifikace | softmax, cross-entropy, dávkový trénink |
| 3    | Vlastní data | normalizace, early stopping, evaluace |
