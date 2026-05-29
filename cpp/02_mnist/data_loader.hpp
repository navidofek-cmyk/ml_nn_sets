#pragma once
// MNIST data loader — čte originální binární formát IDX
// Stáhni data pomocí download_mnist.sh

#include "../common/matrix.hpp"
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <iostream>

// IDX formát je big-endian — převedeme na little-endian pokud jsme na x86
static uint32_t read_uint32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t)bytes[0] << 24 | (uint32_t)bytes[1] << 16
         | (uint32_t)bytes[2] <<  8 | (uint32_t)bytes[3];
}

struct MNISTData {
    Matrix X; // (784 × N) — každý sloupec je jeden obrázek (28×28 pixelů, normalizováno 0–1)
    Matrix Y; // (10 × N)  — one-hot kódování třídy (0–9)
    int N;    // počet vzorků
};

// Načti obrázky z IDX souboru
// Vrátí matici (784 × N), hodnoty normalizované na [0, 1]
Matrix load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Nelze otevřít: " + path);

    uint32_t magic = read_uint32(f);
    if (magic != 2051) throw std::runtime_error("Špatný magic number v: " + path);

    uint32_t N    = read_uint32(f);
    uint32_t rows = read_uint32(f);
    uint32_t cols = read_uint32(f);

    std::cout << "Načítám obrázky: " << N << " × " << rows << "×" << cols << "\n";

    Matrix X(rows * cols, N);
    std::vector<uint8_t> buf(rows * cols);
    for (uint32_t n = 0; n < N; n++) {
        f.read(reinterpret_cast<char*>(buf.data()), rows * cols);
        for (uint32_t i = 0; i < rows * cols; i++)
            X.at(i, n) = buf[i] / 255.0; // normalizace
    }
    return X;
}

// Načti štítky a převeď na one-hot matici (10 × N)
Matrix load_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Nelze otevřít: " + path);

    uint32_t magic = read_uint32(f);
    if (magic != 2049) throw std::runtime_error("Špatný magic number v: " + path);

    uint32_t N = read_uint32(f);
    std::cout << "Načítám štítky: " << N << "\n";

    Matrix Y(10, N, 0.0); // one-hot, 10 tříd
    for (uint32_t n = 0; n < N; n++) {
        uint8_t label;
        f.read(reinterpret_cast<char*>(&label), 1);
        Y.at(label, n) = 1.0; // nastavíme jedničku na správnou třídu
    }
    return Y;
}

MNISTData load_mnist(const std::string& data_dir) {
    MNISTData d;
    d.X = load_images(data_dir + "/train-images-idx3-ubyte");
    d.Y = load_labels(data_dir + "/train-labels-idx1-ubyte");
    d.N = d.X.cols;
    return d;
}

MNISTData load_mnist_test(const std::string& data_dir) {
    MNISTData d;
    d.X = load_images(data_dir + "/t10k-images-idx3-ubyte");
    d.Y = load_labels(data_dir + "/t10k-labels-idx1-ubyte");
    d.N = d.X.cols;
    return d;
}
