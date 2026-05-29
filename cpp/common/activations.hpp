#pragma once
// Aktivační funkce a jejich derivace.
// Každá funkce pracuje element-wise na matrici.

#include "matrix.hpp"
#include <cmath>
#include <algorithm>

// ── ReLU ────────────────────────────────────────────────────────────────────
// f(z) = max(0, z)
inline Matrix relu(const Matrix& z) {
    return z.apply([](double v) { return std::max(0.0, v); });
}

// f'(z) = 1 pokud z > 0, jinak 0
inline Matrix relu_deriv(const Matrix& z) {
    return z.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; });
}

// ── Sigmoid ──────────────────────────────────────────────────────────────────
// f(z) = 1 / (1 + e^(-z))
inline Matrix sigmoid(const Matrix& z) {
    return z.apply([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
}

// f'(z) = f(z) * (1 - f(z))
inline Matrix sigmoid_deriv(const Matrix& z) {
    return z.apply([](double v) {
        double s = 1.0 / (1.0 + std::exp(-v));
        return s * (1.0 - s);
    });
}

// ── Tanh ─────────────────────────────────────────────────────────────────────
inline Matrix tanh_act(const Matrix& z) {
    return z.apply([](double v) { return std::tanh(v); });
}

// f'(z) = 1 - tanh²(z)
inline Matrix tanh_deriv(const Matrix& z) {
    return z.apply([](double v) {
        double t = std::tanh(v);
        return 1.0 - t * t;
    });
}

// ── Lineární (identita) ──────────────────────────────────────────────────────
// Pro výstupní vrstvu regrese
inline Matrix linear(const Matrix& z) { return z; }
inline Matrix linear_deriv(const Matrix& z) {
    return Matrix(z.rows, z.cols, 1.0);
}

// ── Softmax ───────────────────────────────────────────────────────────────────
// Aplikuje se po sloupcích (každý sloupec = jeden vzorek)
// f(z_i) = e^(z_i - max) / Σ e^(z_j - max)   ← numerická stabilita
inline Matrix softmax(const Matrix& z) {
    Matrix result(z.rows, z.cols);
    for (int j = 0; j < z.cols; j++) {
        // najdi maximum v sloupci (numerická stabilita)
        double maxv = z.at(0, j);
        for (int i = 1; i < z.rows; i++) maxv = std::max(maxv, z.at(i, j));

        double sumexp = 0.0;
        for (int i = 0; i < z.rows; i++) sumexp += std::exp(z.at(i, j) - maxv);

        for (int i = 0; i < z.rows; i++)
            result.at(i, j) = std::exp(z.at(i, j) - maxv) / sumexp;
    }
    return result;
}
// Pozn.: derivace softmax se nekombinuje se softmax_deriv zvlášť —
// při použití cross-entropy loss se gradient zjednodušší na (ŷ - y).
// Viz losses.hpp.

// ── Enum pro výběr aktivace ───────────────────────────────────────────────────
enum class Activation { ReLU, Sigmoid, Tanh, Linear, Softmax };

inline Matrix apply_activation(const Matrix& z, Activation act) {
    switch (act) {
        case Activation::ReLU:    return relu(z);
        case Activation::Sigmoid: return sigmoid(z);
        case Activation::Tanh:    return tanh_act(z);
        case Activation::Linear:  return linear(z);
        case Activation::Softmax: return softmax(z);
    }
    return z;
}

// Vrátí derivaci aktivace vyhodnocenou v z
inline Matrix apply_activation_deriv(const Matrix& z, Activation act) {
    switch (act) {
        case Activation::ReLU:    return relu_deriv(z);
        case Activation::Sigmoid: return sigmoid_deriv(z);
        case Activation::Tanh:    return tanh_deriv(z);
        case Activation::Linear:  return linear_deriv(z);
        case Activation::Softmax: return Matrix(z.rows, z.cols, 1.0); // handled separately
    }
    return Matrix(z.rows, z.cols, 1.0);
}
