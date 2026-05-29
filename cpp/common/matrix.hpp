#pragma once
// Jednoduchá 2D matice — základ pro veškeré výpočty v síti.
// Uložení: row-major (data[r * cols + c] = prvek na řádku r, sloupci c)

#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <iostream>
#include <functional>
#include <stdexcept>

class Matrix {
public:
    int rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int rows, int cols, double val = 0.0)
        : rows(rows), cols(cols), data(rows * cols, val) {}

    double& at(int r, int c) { return data[r * cols + c]; }
    const double& at(int r, int c) const { return data[r * cols + c]; }

    // Maticové násobení: (this: m×k) * (other: k×n) → (m×n)
    Matrix operator*(const Matrix& other) const {
        assert(cols == other.rows && "Rozměry matic nesedí pro násobení");
        Matrix result(rows, other.cols, 0.0);
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < cols; k++)
                for (int j = 0; j < other.cols; j++)
                    result.at(i, j) += at(i, k) * other.at(k, j);
        return result;
    }

    // Sčítání po prvcích
    Matrix operator+(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); i++)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    // Odečítání po prvcích
    Matrix operator-(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); i++)
            result.data[i] = data[i] - other.data[i];
        return result;
    }

    // Násobení skalárem
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); i++)
            result.data[i] = data[i] * scalar;
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols);
        for (int i = 0; i < (int)data.size(); i++) data[i] += other.data[i];
        return *this;
    }

    // Hadamardův součin (násobení po prvcích): a ⊙ b
    Matrix hadamard(const Matrix& other) const {
        assert(rows == other.rows && cols == other.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); i++)
            result.data[i] = data[i] * other.data[i];
        return result;
    }

    // Transponování
    Matrix T() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.at(j, i) = at(i, j);
        return result;
    }

    // Aplikuj funkci na každý prvek
    Matrix apply(std::function<double(double)> f) const {
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); i++)
            result.data[i] = f(data[i]);
        return result;
    }

    // Přičti bias k každému sloupci (bias má rows=n, cols=1)
    // Vstup: this=(n×B), bias=(n×1) → výsledek=(n×B)
    Matrix add_bias(const Matrix& bias) const {
        assert(rows == bias.rows && bias.cols == 1);
        Matrix result(rows, cols);
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < rows; i++)
                result.at(i, j) = at(i, j) + bias.at(i, 0);
        return result;
    }

    // Sečti všechny sloupce → sloupcový vektor (n×1)
    Matrix sum_cols() const {
        Matrix result(rows, 1, 0.0);
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < rows; i++)
                result.at(i, 0) += at(i, j);
        return result;
    }

    // Průměr všech prvků
    double mean() const {
        double s = 0;
        for (double v : data) s += v;
        return s / data.size();
    }

    // Xavier inicializace: vhodné pro sigmoid/tanh
    void xavier_init(int fan_in, int fan_out, unsigned seed = 42) {
        double limit = std::sqrt(6.0 / (fan_in + fan_out));
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (auto& v : data) v = dist(rng);
    }

    // He inicializace: vhodné pro ReLU
    void he_init(int fan_in, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / fan_in));
        for (auto& v : data) v = dist(rng);
    }

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " (" << rows << "x" << cols << "):\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                std::cout << at(i, j) << " ";
            std::cout << "\n";
        }
    }
};

inline Matrix operator*(double scalar, const Matrix& m) { return m * scalar; }
