// ============================================================
// ČÁST 1: Neuronová síť pro regresi
// ============================================================
// Úloha: Naučit síť funkci y = sin(x) + šum
//
// Architektura:
//   vstup(1) → Dense(32, ReLU) → Dense(32, ReLU) → výstup(1, Linear)
//
// Loss: MSE
// Optimizer: SGD
// ============================================================

#include "../common/matrix.hpp"
#include "../common/activations.hpp"
#include "../common/losses.hpp"
#include "../common/network.hpp"

#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>

// Generuj trénovací data: y = sin(2πx) + šum
// X: (1 × N), Y: (1 × N)
std::pair<Matrix, Matrix> generate_data(int N, double noise_std = 0.1, unsigned seed = 0) {
    Matrix X(1, N), Y(1, N);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
    std::normal_distribution<double> noise(0.0, noise_std);

    for (int i = 0; i < N; i++) {
        double x = x_dist(rng);
        X.at(0, i) = x;
        Y.at(0, i) = std::sin(2.0 * M_PI * x) + noise(rng);
    }
    return {X, Y};
}

int main() {
    std::cout << "=== Část 1: Regrese (sin funkce) ===\n\n";

    // ── Data ──────────────────────────────────────────────────
    int N_train = 1000, N_test = 200;
    auto [X_train, Y_train] = generate_data(N_train, 0.1, 42);
    auto [X_test, Y_test]   = generate_data(N_test,  0.1, 99);

    std::cout << "Trénovací vzorky: " << N_train << "\n";
    std::cout << "Testovací vzorky: " << N_test  << "\n\n";

    // ── Síť ───────────────────────────────────────────────────
    // learning_rate = 0.01
    Network net(0.01);
    net.add_layer(1,  32, Activation::ReLU,   42);   // vstup(1) → 32 neuronů
    net.add_layer(32, 32, Activation::ReLU,   43);   // 32 → 32
    net.add_layer(32,  1, Activation::Linear, 44);   // 32 → výstup(1)
    net.summary();

    // ── Trénink ───────────────────────────────────────────────
    int epochs     = 500;
    int batch_size = 64;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Epocha | Train MSE  | Test MSE\n";
    std::cout << "-------|------------|----------\n";

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double train_loss = net.train_epoch(
            X_train, Y_train, batch_size,
            mse_loss,   // loss funkce
            mse_grad    // gradient loss vůči ŷ
        );

        if (epoch % 50 == 0) {
            Matrix y_hat_test = net.forward(X_test);
            double test_loss = mse_loss(y_hat_test, Y_test);
            std::cout << std::setw(6) << epoch << " | "
                      << std::setw(10) << train_loss << " | "
                      << std::setw(8) << test_loss << "\n";
        }
    }

    // ── Ukázka predikcí ───────────────────────────────────────
    std::cout << "\nUkázka predikcí vs. skutečnost:\n";
    std::cout << "     x    |   y_true  |   y_pred\n";
    std::cout << "----------|-----------|----------\n";

    // Vyber 10 testovacích vzorků
    Matrix X_sample(1, 10), Y_sample(1, 10);
    for (int i = 0; i < 10; i++) {
        X_sample.at(0, i) = X_test.at(0, i);
        Y_sample.at(0, i) = Y_test.at(0, i);
    }
    Matrix Y_pred = net.forward(X_sample);

    for (int i = 0; i < 10; i++) {
        std::cout << std::setw(9) << X_sample.at(0, i) << " | "
                  << std::setw(9) << Y_sample.at(0, i) << " | "
                  << std::setw(9) << Y_pred.at(0, i)  << "\n";
    }

    net.save("../../weights/regression.bin");
    std::cout << "\nHotovo!\n";
    return 0;
}

// ============================================================
// CO SE TADY DĚJE (krok po kroku):
//
// 1. Generujeme 1000 bodů (x, sin(2πx)+šum) jako trénovací data
//
// 2. Síť: 1 → 32 → 32 → 1
//    - Dvě skryté vrstvy s ReLU (přidávají nelinearitu)
//    - Výstupní vrstva bez aktivace (linear) → výstup může být libovolné číslo
//
// 3. Trénink:
//    a) forward(X_batch)    → dostaneme y_hat
//    b) mse_loss(y_hat, Y)  → číslo = jak moc se mýlíme
//    c) mse_grad(y_hat, Y)  → gradient, který říká "kam jít"
//    d) backward(X, grad)   → spočítáme ∂L/∂W pro každou vrstvu
//    e) update()            → W -= lr * ∂L/∂W
//
// 4. Opakujeme 500 epoch, loss by měla klesat
// ============================================================
