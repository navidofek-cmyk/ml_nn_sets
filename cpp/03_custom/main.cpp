// ============================================================
// ČÁST 3: Vlastní dataset
// ============================================================
// Šablona pro vlastní data + pokročilé techniky:
//   - Normalizace vstupu (z-score)
//   - Early stopping
//   - Evaluace (precision, recall pro binární klasifikaci)
//   - CSV loader
//
// Příklad: binární klasifikace (spirála 2 tříd)
// Architektura: vstup(2) → 64 → 64 → výstup(1, Sigmoid)
// Loss: Binary Cross-Entropy
// ============================================================

#include "../common/matrix.hpp"
#include "../common/activations.hpp"
#include "../common/losses.hpp"
#include "../common/network.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <numeric>

// ── Normalizace (z-score): x' = (x - μ) / σ ─────────────────────────────────
// Vrátí {mean_vec, std_vec} pro každý feature (řádek matice X)
std::pair<std::vector<double>, std::vector<double>> compute_stats(const Matrix& X) {
    int F = X.rows, N = X.cols;
    std::vector<double> mean(F, 0.0), stddev(F, 0.0);
    for (int i = 0; i < F; i++) {
        for (int j = 0; j < N; j++) mean[i] += X.at(i, j);
        mean[i] /= N;
        for (int j = 0; j < N; j++) {
            double d = X.at(i, j) - mean[i];
            stddev[i] += d * d;
        }
        stddev[i] = std::sqrt(stddev[i] / N + 1e-8); // +ε pro numerickou stabilitu
    }
    return {mean, stddev};
}

Matrix normalize(const Matrix& X,
                 const std::vector<double>& mean,
                 const std::vector<double>& stddev) {
    Matrix result(X.rows, X.cols);
    for (int i = 0; i < X.rows; i++)
        for (int j = 0; j < X.cols; j++)
            result.at(i, j) = (X.at(i, j) - mean[i]) / stddev[i];
    return result;
}

// ── Two-moons dataset — klasický benchmark pro binární klasifikaci ────────────
// Třída 0: horní půlkruh  Třída 1: dolní půlkruh posunutý doprava a dolů
// Lineárně NESEPAROVATELNÉ, ale snadno naučitelné pro NN.
std::pair<Matrix, Matrix> generate_moons(int N_per_class, unsigned seed = 42) {
    int N = N_per_class * 2;
    Matrix X(2, N), Y(1, N, 0.0);
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.15);

    for (int i = 0; i < N_per_class; i++) {
        // Třída 0: horní oblouk, angle ∈ [0, π]
        double angle = M_PI * i / (N_per_class - 1);
        X.at(0, i) = std::cos(angle) + noise(rng);
        X.at(1, i) = std::sin(angle) + noise(rng);
        Y.at(0, i) = 0.0;
    }
    for (int i = 0; i < N_per_class; i++) {
        // Třída 1: dolní oblouk, angle ∈ [π, 2π], posunutý o (1, -0.5)
        double angle = M_PI + M_PI * i / (N_per_class - 1);
        X.at(0, N_per_class + i) = std::cos(angle) + 1.0 + noise(rng);
        X.at(1, N_per_class + i) = std::sin(angle) - 0.5 + noise(rng);
        Y.at(0, N_per_class + i) = 1.0;
    }
    return {X, Y};
}

// ── Zamíchání dat ─────────────────────────────────────────────────────────────
void shuffle(Matrix& X, Matrix& Y, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<int> idx(X.cols);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    Matrix X2(X.rows, X.cols), Y2(Y.rows, Y.cols);
    for (int j = 0; j < X.cols; j++) {
        for (int i = 0; i < X.rows; i++) X2.at(i, j) = X.at(i, idx[j]);
        for (int i = 0; i < Y.rows; i++) Y2.at(i, j) = Y.at(i, idx[j]);
    }
    X = X2; Y = Y2;
}

// ── Evaluace binární klasifikace ─────────────────────────────────────────────
struct BinaryMetrics {
    double accuracy, precision, recall, f1;
};

BinaryMetrics evaluate_binary(const Matrix& y_hat, const Matrix& y, double threshold = 0.5) {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    for (int j = 0; j < y_hat.cols; j++) {
        int pred  = y_hat.at(0, j) >= threshold ? 1 : 0;
        int label = (int)y.at(0, j);
        if (pred == 1 && label == 1) tp++;
        else if (pred == 1 && label == 0) fp++;
        else if (pred == 0 && label == 1) fn++;
        else tn++;
    }
    double acc  = (double)(tp + tn) / y_hat.cols;
    double prec = tp + fp > 0 ? (double)tp / (tp + fp) : 0.0;
    double rec  = tp + fn > 0 ? (double)tp / (tp + fn) : 0.0;
    double f1   = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
    return {acc, prec, rec, f1};
}

int main() {
    std::cout << "=== Část 3: Vlastní dataset (two-moons, binární klasifikace) ===\n\n";

    // ── Data ──────────────────────────────────────────────────
    int N_per_class = 500;
    auto [X_all, Y_all] = generate_moons(N_per_class, 42);
    shuffle(X_all, Y_all, 0);

    // Train/val split: 80% / 20%
    int N_total = X_all.cols;
    int N_train = (int)(N_total * 0.8);
    int N_val   = N_total - N_train;

    Matrix X_train(2, N_train), Y_train(1, N_train);
    Matrix X_val(2, N_val),   Y_val(1, N_val);

    for (int j = 0; j < N_train; j++) {
        for (int i = 0; i < 2; i++) X_train.at(i, j) = X_all.at(i, j);
        Y_train.at(0, j) = Y_all.at(0, j);
    }
    for (int j = 0; j < N_val; j++) {
        for (int i = 0; i < 2; i++) X_val.at(i, j) = X_all.at(i, N_train + j);
        Y_val.at(0, j) = Y_all.at(0, N_train + j);
    }

    // Normalizace (počítáme statistiky pouze na trénovací sadě!)
    auto [mean, stddev] = compute_stats(X_train);
    X_train = normalize(X_train, mean, stddev);
    X_val   = normalize(X_val,   mean, stddev);

    std::cout << "Trénovací vzorky: " << N_train << "\n";
    std::cout << "Validační vzorky: " << N_val   << "\n\n";

    // ── Síť ───────────────────────────────────────────────────
    // Adam: lr=0.001 je výchozí, konverguje rychleji než SGD na komplexních datech
    Network net(0.001, Optimizer::Adam);
    net.add_layer(2,  64, Activation::ReLU,    42);
    net.add_layer(64, 64, Activation::ReLU,    43);
    net.add_layer(64,  1, Activation::Sigmoid, 44);
    net.summary();

    // ── Early stopping ────────────────────────────────────────
    int    patience    = 100;     // zastavíme pokud se val_loss nezlepší 100 epoch
    double best_val    = 1e9;
    int    no_improve  = 0;
    int    best_epoch  = 0;

    int epochs     = 1000;
    int batch_size = 64;

    std::cout << "Epocha | Train BCE  | Train Acc  | Val BCE    | Val Acc\n";
    std::cout << "-------|------------|------------|------------|--------\n";
    std::cout << std::fixed << std::setprecision(4);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        shuffle(X_train, Y_train, epoch);

        double train_loss = net.train_epoch(
            X_train, Y_train, batch_size,
            bce_loss,          // Binary Cross-Entropy
            bce_sigmoid_grad,  // Kombinovaný gradient sigmoid+BCE = (ŷ-y)/n
            true               // combined=true: backprop nepřidá znovu sigmoid'(z)
        );

        Matrix y_hat_train = net.forward(X_train);
        Matrix y_hat_val   = net.forward(X_val);
        double val_loss    = bce_loss(y_hat_val, Y_val);
        auto   tr_met      = evaluate_binary(y_hat_train, Y_train);
        auto   val_met     = evaluate_binary(y_hat_val,   Y_val);

        if (epoch % 50 == 0 || epoch == 1) {
            std::cout << std::setw(6) << epoch << " | "
                      << std::setw(10) << train_loss        << " | "
                      << std::setw(9)  << tr_met.accuracy * 100 << "% | "
                      << std::setw(10) << val_loss          << " | "
                      << std::setw(6)  << val_met.accuracy * 100 << "%\n";
        }

        // Early stopping
        if (val_loss < best_val - 1e-5) {
            best_val   = val_loss;
            best_epoch = epoch;
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= patience) {
                std::cout << "\nEarly stopping v epoše " << epoch
                          << " (nejlepší epocha: " << best_epoch << ")\n";
                break;
            }
        }
    }

    // ── Finální evaluace ──────────────────────────────────────
    Matrix y_hat_val = net.forward(X_val);
    auto m = evaluate_binary(y_hat_val, Y_val);
    std::cout << "\n=== Výsledky na validační sadě ===\n";
    std::cout << "Accuracy:  " << m.accuracy  * 100 << "%\n";
    std::cout << "Precision: " << m.precision * 100 << "%\n";
    std::cout << "Recall:    " << m.recall    * 100 << "%\n";
    std::cout << "F1 score:  " << m.f1        * 100 << "%\n";

    net.save("../../weights/moons.bin");
    return 0;
}

// ============================================================
// KLÍČOVÉ KONCEPTY v této části:
//
// 1. Normalizace vstupů (z-score):
//    - NN funguje špatně pokud mají features různé škály
//    - Vždy počítej μ a σ POUZE z trénovacích dat!
//    - Test/val data normalizuj stejnými hodnotami
//
// 2. Train/val split:
//    - Val sada = data která síť "nevidí" při tréninku
//    - Slouží k detekci overfittingu
//
// 3. Early stopping:
//    - Sleduj val_loss; pokud se N epoch nezlepšuje → zastav
//    - Jednoduchá regularizace (zabrání přetrénování)
//
// 4. Binární metriky:
//    - Accuracy nestačí (nevyvážené třídy)
//    - Precision = TP / (TP + FP)   — jak přesné jsou naše pozitivní predikce
//    - Recall    = TP / (TP + FN)   — kolik pozitivních jsme zachytili
//    - F1        = 2·P·R / (P+R)    — harmonický průměr precision a recall
//
// 5. Jak přidat vlastní CSV data:
//    Načti CSV → uložíš do Matrix X(features, N), Y(outputs, N)
//    Pak zbytek kódu funguje stejně.
// ============================================================
