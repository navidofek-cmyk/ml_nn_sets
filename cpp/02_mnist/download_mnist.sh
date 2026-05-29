#!/bin/bash
# Stáhne MNIST dataset do aktuální složky
set -e

BASE="http://yann.lecun.com/exdb/mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for f in "${FILES[@]}"; do
    if [ ! -f "${f%.gz}" ]; then
        echo "Stahuji $f ..."
        curl -O "$BASE/$f" || wget "$BASE/$f"
        gunzip "$f"
    else
        echo "${f%.gz} již existuje, přeskakuji."
    fi
done
echo "Hotovo! MNIST data jsou připravena."
