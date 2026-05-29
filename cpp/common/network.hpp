#pragma once
// Neuronová síť — forward pass, backpropagation, SGD + Adam.
//
// Každá vrstva ukládá:
//   W, b        — parametry
//   dW, db      — gradienty (spočítané v backward)
//   z, a        — mezihodoty pro backprop
//   mW,vW,mb,vb — Adam moment vektory (1. a 2. moment)

#include "matrix.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>

enum class Optimizer { SGD, Adam };

struct Layer {
    Matrix W, b;           // parametry
    Matrix dW, db;         // gradienty
    Matrix z, a;           // cache pro backprop
    Matrix mW, vW, mb, vb; // Adam: 1. a 2. moment pro W a b
    Activation act;
    int in_size, out_size;

    Layer(int in, int out, Activation act, unsigned seed = 42)
        : W(out, in), b(out, 1, 0.0),
          dW(out, in, 0.0), db(out, 1, 0.0),
          mW(out, in, 0.0), vW(out, in, 0.0),
          mb(out, 1, 0.0),  vb(out, 1, 0.0),
          act(act), in_size(in), out_size(out)
    {
        if (act == Activation::ReLU)
            W.he_init(in, seed);
        else
            W.xavier_init(in, out, seed);
    }
};

class Network {
public:
    std::vector<Layer> layers;
    double lr;              // learning rate
    Optimizer opt;          // zvolený optimizer
    int    adam_step;       // čítač kroků pro bias correction
    double adam_b1, adam_b2, adam_eps; // hyperparametry Adama

    Network(double learning_rate = 0.01,
            Optimizer optimizer = Optimizer::SGD)
        : lr(learning_rate), opt(optimizer), adam_step(0),
          adam_b1(0.9), adam_b2(0.999), adam_eps(1e-8) {}

    void add_layer(int in, int out, Activation act, unsigned seed = 42) {
        layers.emplace_back(in, out, act, seed);
    }

    // ── Forward pass ──────────────────────────────────────────────────────────
    Matrix forward(const Matrix& x) {
        Matrix a = x;
        for (auto& layer : layers) {
            layer.z = layer.W * a;
            layer.z = layer.z.add_bias(layer.b);
            layer.a = apply_activation(layer.z, layer.act);
            a = layer.a;
        }
        return a;
    }

    // ── Backward pass (backpropagation) ──────────────────────────────────────
    // combined=true: grad_output je ∂L/∂z (přeskočí aktivaci v poslední vrstvě)
    //   → používej pro softmax+CE nebo sigmoid+BCE
    // combined=false: grad_output je ∂L/∂a (backprop si aplikuje f'(z) sám)
    //   → používej pro mse_grad s lineárním/relu výstupem
    void backward(const Matrix& x, const Matrix& grad_output, bool combined = false) {
        int L = layers.size();
        Matrix delta = grad_output;

        for (int l = L - 1; l >= 0; l--) {
            Layer& layer = layers[l];
            const Matrix& a_prev = (l == 0) ? x : layers[l - 1].a;

            bool skip = (layer.act == Activation::Softmax) ||
                        (combined && l == L - 1);
            if (!skip)
                delta = delta.hadamard(apply_activation_deriv(layer.z, layer.act));

            // ∂L/∂W: delta má 1/B od grad_fn, součet přes B sloupců dá průměr
            layer.dW = delta * a_prev.T();
            // ∂L/∂b: sum_cols při 1/B delta dá průměr (NEDělíme znovu B)
            layer.db = delta.sum_cols();

            if (l > 0) delta = layer.W.T() * delta;
        }
    }

    // ── SGD update ────────────────────────────────────────────────────────────
    void update_sgd() {
        for (auto& l : layers) {
            l.W = l.W - l.dW * lr;
            l.b = l.b - l.db * lr;
        }
    }

    // ── Adam update ───────────────────────────────────────────────────────────
    // Adam (Kingma & Ba, 2015):
    //   m = β₁·m + (1-β₁)·g           ← 1. moment (klouzavý průměr gradientu)
    //   v = β₂·v + (1-β₂)·g²          ← 2. moment (klouzavý průměr g²)
    //   m̂ = m / (1 - β₁^t)            ← bias correction
    //   v̂ = v / (1 - β₂^t)
    //   θ = θ - α · m̂ / (√v̂ + ε)
    void update_adam() {
        adam_step++;
        double bc1 = 1.0 - std::pow(adam_b1, adam_step); // 1 - β₁^t
        double bc2 = 1.0 - std::pow(adam_b2, adam_step); // 1 - β₂^t

        for (auto& l : layers) {
            int nW = l.W.rows * l.W.cols;
            for (int i = 0; i < nW; i++) {
                double g = l.dW.data[i];
                l.mW.data[i] = adam_b1 * l.mW.data[i] + (1 - adam_b1) * g;
                l.vW.data[i] = adam_b2 * l.vW.data[i] + (1 - adam_b2) * g * g;
                double mh = l.mW.data[i] / bc1;
                double vh = l.vW.data[i] / bc2;
                l.W.data[i] -= lr * mh / (std::sqrt(vh) + adam_eps);
            }
            int nb = l.b.rows;
            for (int i = 0; i < nb; i++) {
                double g = l.db.data[i];
                l.mb.data[i] = adam_b1 * l.mb.data[i] + (1 - adam_b1) * g;
                l.vb.data[i] = adam_b2 * l.vb.data[i] + (1 - adam_b2) * g * g;
                double mh = l.mb.data[i] / bc1;
                double vh = l.vb.data[i] / bc2;
                l.b.data[i] -= lr * mh / (std::sqrt(vh) + adam_eps);
            }
        }
    }

    void update() {
        if (opt == Optimizer::Adam) update_adam();
        else                        update_sgd();
    }

    // ── Trénink jedné epochy ───────────────────────────────────────────────────
    // combined: true pokud grad_fn vrací ∂L/∂z (softmax+CE, sigmoid+BCE)
    template<typename LossFn, typename GradFn>
    double train_epoch(const Matrix& X, const Matrix& Y,
                       int batch_size,
                       LossFn loss_fn, GradFn grad_fn,
                       bool combined = false) {
        int N = X.cols;
        double total_loss = 0.0;
        int num_batches = 0;

        for (int start = 0; start < N; start += batch_size) {
            int end = std::min(start + batch_size, N);
            int B = end - start;

            Matrix X_batch(X.rows, B), Y_batch(Y.rows, B);
            for (int j = 0; j < B; j++) {
                for (int i = 0; i < X.rows; i++) X_batch.at(i, j) = X.at(i, start + j);
                for (int i = 0; i < Y.rows; i++) Y_batch.at(i, j) = Y.at(i, start + j);
            }

            Matrix y_hat = forward(X_batch);
            total_loss += loss_fn(y_hat, Y_batch);
            backward(X_batch, grad_fn(y_hat, Y_batch), combined);
            update();
            num_batches++;
        }
        return total_loss / num_batches;
    }

    // ── Shrnutí architektury ───────────────────────────────────────────────────
    void summary() const {
        std::cout << "\n=== Architektura sítě ===\n";
        long long total_params = 0;
        for (int i = 0; i < (int)layers.size(); i++) {
            const auto& l = layers[i];
            long long params = (long long)l.W.rows * l.W.cols + l.b.rows;
            total_params += params;
            std::string act_name;
            switch (l.act) {
                case Activation::ReLU:    act_name = "ReLU";    break;
                case Activation::Sigmoid: act_name = "Sigmoid"; break;
                case Activation::Tanh:    act_name = "Tanh";    break;
                case Activation::Linear:  act_name = "Linear";  break;
                case Activation::Softmax: act_name = "Softmax"; break;
            }
            std::cout << "Vrstva " << i+1 << ": "
                      << l.in_size << " → " << l.out_size
                      << "  [" << act_name << "]"
                      << "  params=" << params << "\n";
        }
        std::string opt_name = (opt == Optimizer::Adam) ? "Adam" : "SGD";
        std::cout << "Celkem parametrů: " << total_params << "\n";
        std::cout << "Optimizer: " << opt_name << "  lr=" << lr << "\n\n";
    }
};
