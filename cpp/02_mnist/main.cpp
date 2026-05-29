// ============================================================
// ČÁST 2: Klasifikace ručně psaných číslic (MNIST)
// ============================================================
// Úloha: Rozpoznat číslice 0–9 ze 28×28 pixelů
//
// Vstup:  784 pixelů (28×28, normalizované 0–1)
// Výstup: 10 tříd (softmax pravděpodobnosti)
//
// Architektura:
//   vstup(784) → Dense(256, ReLU) → Dense(128, ReLU) → výstup(10, Softmax)
//
// Loss: Cross-Entropy
// Optimizer: SGD
//
// Stáhni data: bash download_mnist.sh
// ============================================================

#include "../common/matrix.hpp"
#include "../common/activations.hpp"
#include "../common/losses.hpp"
#include "../common/network.hpp"
#include "data_loader.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>

// Zamíchá sloupce matic X a Y synchronně (pro stochastický trénink)
void shuffle_data(Matrix& X, Matrix& Y, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<int> idx(X.cols);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    Matrix X2(X.rows, X.cols), Y2(Y.rows, Y.cols);
    for (int j = 0; j < X.cols; j++) {
        int src = idx[j];
        for (int i = 0; i < X.rows; i++) X2.at(i, j) = X.at(i, src);
        for (int i = 0; i < Y.rows; i++) Y2.at(i, j) = Y.at(i, src);
    }
    X = X2; Y = Y2;
}

int main(int argc, char* argv[]) {
    std::string data_dir = ".";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== Část 2: MNIST klasifikace ===\n\n";

    // ── Data ──────────────────────────────────────────────────
    MNISTData train, test;
    try {
        train = load_mnist(data_dir);
        test  = load_mnist_test(data_dir);
    } catch (const std::exception& e) {
        std::cerr << "\nChyba: " << e.what() << "\n";
        std::cerr << "Stáhni data: cd cpp/02_mnist && bash download_mnist.sh\n";
        return 1;
    }

    std::cout << "\nTrénovací vzorky: " << train.N << "\n";
    std::cout << "Testovací vzorky: " << test.N  << "\n\n";

    // ── Síť ───────────────────────────────────────────────────
    Network net(0.1); // vyšší lr — SGD na MNIST konverguje rychle
    net.add_layer(784, 256, Activation::ReLU,    42);
    net.add_layer(256, 128, Activation::ReLU,    43);
    net.add_layer(128,  10, Activation::Softmax, 44);
    net.summary();

    // ── Trénink ───────────────────────────────────────────────
    int epochs     = 20;
    int batch_size = 128;

    std::cout << "Epocha | Train CE   | Train Acc  | Test Acc\n";
    std::cout << "-------|------------|------------|----------\n";
    std::cout << std::fixed << std::setprecision(4);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        auto t0 = std::chrono::steady_clock::now();

        // Zamíchej data před každou epochou
        shuffle_data(train.X, train.Y, epoch);

        double train_loss = net.train_epoch(
            train.X, train.Y, batch_size,
            cross_entropy_loss,        // L = -Σ y·log(ŷ)
            softmax_crossentropy_grad  // ∂L/∂z = (ŷ - y)/B  ← kombinovaný gradient
        );

        // Výpočet přesnosti
        Matrix y_hat_train = net.forward(train.X);
        Matrix y_hat_test  = net.forward(test.X);
        double train_acc   = accuracy(y_hat_train, train.Y);
        double test_acc    = accuracy(y_hat_test,  test.Y);

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::cout << std::setw(6) << epoch << " | "
                  << std::setw(10) << train_loss << " | "
                  << std::setw(9)  << train_acc * 100 << "% | "
                  << std::setw(7)  << test_acc * 100 << "%"
                  << "  (" << std::setprecision(1) << secs << "s)\n";
        std::cout << std::setprecision(4);
    }

    // ── Výsledek ──────────────────────────────────────────────
    Matrix final_pred = net.forward(test.X);
    double final_acc  = accuracy(final_pred, test.Y);
    std::cout << "\nFinální testovací přesnost: "
              << std::setprecision(2) << final_acc * 100 << "%\n";
    std::cout << "(Benchmark: ~97-98% s jednoduchou MLP)\n\n";

    return 0;
}

// ============================================================
// KLÍČOVÉ ROZDÍLY oproti regresi:
//
// 1. One-hot kódování: štítek "3" → [0,0,0,1,0,0,0,0,0,0]
//
// 2. Softmax výstup: 10 čísel, každé ∈ (0,1), součet = 1
//    → interpretujeme jako pravděpodobnosti tříd
//
// 3. Cross-entropy loss místo MSE:
//    L = -log(ŷ_správná_třída)
//    → penalizuje sebevědomí v špatné třídě logaritmicky
//
// 4. Kombinovaný gradient softmax+CE:
//    ∂L/∂z = ŷ - y  (kde y je one-hot)
//    Toto je numericky stabilnější a jednodušší než oddělené derivace
//
// 5. Míchání dat před každou epochou = lepší generalizace
// ============================================================
