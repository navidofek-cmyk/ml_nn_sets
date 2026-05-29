#pragma once
// Ztrátové funkce a jejich gradienty vůči výstupu sítě (∂L/∂ŷ).

#include "matrix.hpp"
#include <cmath>
#include <stdexcept>

// ── MSE (Mean Squared Error) — pro regresi ────────────────────────────────────
// L = (1/n) · Σ (ŷ_i - y_i)²
// kde n = počet prvků (rows * batch_size)
inline double mse_loss(const Matrix& y_hat, const Matrix& y) {
    assert(y_hat.rows == y.rows && y_hat.cols == y.cols);
    double loss = 0.0;
    int n = y_hat.rows * y_hat.cols;
    for (int i = 0; i < n; i++) {
        double diff = y_hat.data[i] - y.data[i];
        loss += diff * diff;
    }
    return loss / n;
}

// ∂L/∂ŷ = (2/n) · (ŷ - y)
inline Matrix mse_grad(const Matrix& y_hat, const Matrix& y) {
    assert(y_hat.rows == y.rows && y_hat.cols == y.cols);
    int n = y_hat.rows * y_hat.cols;
    return (y_hat - y) * (2.0 / n);
}

// ── Cross-Entropy — pro klasifikaci (se softmaxem) ───────────────────────────
// L = -(1/B) · Σ_i Σ_c y_ic · log(ŷ_ic)
// kde B = batch size, y je one-hot matice
inline double cross_entropy_loss(const Matrix& y_hat, const Matrix& y) {
    assert(y_hat.rows == y.rows && y_hat.cols == y.cols);
    double loss = 0.0;
    int n = y_hat.cols; // batch size
    for (int j = 0; j < y_hat.cols; j++)
        for (int i = 0; i < y_hat.rows; i++)
            if (y.at(i, j) > 0.0)
                loss -= y.at(i, j) * std::log(std::max(y_hat.at(i, j), 1e-15));
    return loss / n;
}

// Kombinovaný gradient softmax + cross-entropy:
// ∂L/∂z = (1/B) · (ŷ - y)
// Toto je přímý gradient vůči z (pre-activation výstupní vrstvy).
// Používáme místo oddělené derivace softmax a cross-entropy.
inline Matrix softmax_crossentropy_grad(const Matrix& y_hat, const Matrix& y) {
    assert(y_hat.rows == y.rows && y_hat.cols == y.cols);
    int B = y_hat.cols;
    return (y_hat - y) * (1.0 / B);
}

// ── Binary Cross-Entropy — pro binární klasifikaci (se sigmoidem) ────────────
// L = -(1/n) · Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
inline double bce_loss(const Matrix& y_hat, const Matrix& y) {
    double loss = 0.0;
    int n = y_hat.rows * y_hat.cols;
    for (int i = 0; i < n; i++) {
        double p = std::max(std::min(y_hat.data[i], 1.0 - 1e-15), 1e-15);
        loss -= y.data[i] * std::log(p) + (1.0 - y.data[i]) * std::log(1.0 - p);
    }
    return loss / n;
}

// ∂L/∂ŷ = (1/n) · (ŷ - y) / (ŷ·(1-ŷ))
// Se sigmoidem se gradient opět zjednodušší na (ŷ - y)/n
inline Matrix bce_sigmoid_grad(const Matrix& y_hat, const Matrix& y) {
    int n = y_hat.rows * y_hat.cols;
    return (y_hat - y) * (1.0 / n);
}

// ── Přesnost (accuracy) ───────────────────────────────────────────────────────
// Předpokládá y_hat = softmax výstup (rows=třídy, cols=vzorky)
// y = one-hot matice stejného tvaru
inline double accuracy(const Matrix& y_hat, const Matrix& y) {
    assert(y_hat.cols == y.cols);
    int correct = 0;
    for (int j = 0; j < y_hat.cols; j++) {
        // najdi argmax v y_hat
        int pred = 0;
        for (int i = 1; i < y_hat.rows; i++)
            if (y_hat.at(i, j) > y_hat.at(pred, j)) pred = i;
        // najdi argmax v y
        int label = 0;
        for (int i = 1; i < y.rows; i++)
            if (y.at(i, j) > y.at(label, j)) label = i;
        if (pred == label) correct++;
    }
    return (double)correct / y_hat.cols;
}
