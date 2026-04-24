// Copyright 2026 BPCells contributors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#pragma once

#include "MatrixIterator.h"

namespace BPCells {

// Element-wise addition of two input matrices with matching dimensions.
template <typename T> class MatrixAddition : public MatrixLoader<T> {
  protected:
    // Running invariants:
    // - left and right are always about to read the next column for output
    // - left and right are always positioned at the same column (lockstep)
    // - row_idx is the next row to resume output from
    std::unique_ptr<MatrixLoader<T>> left, right;
    std::vector<T> col_buffer, val_buffer;
    std::vector<uint32_t> row_buffer;
    uint32_t row_idx = 0;
    uint32_t max_load_size, loaded;

    void accumulate_column(MatrixLoader<T> &m) {
        while (m.load()) {
            uint32_t cap = m.capacity();
            const uint32_t *rd = m.rowData();
            const T *vd = m.valData();
            for (uint32_t i = 0; i < cap; i++) {
                col_buffer[rd[i]] += vd[i];
            }
        }
    }

  public:
    MatrixAddition(std::unique_ptr<MatrixLoader<T>> &&left, std::unique_ptr<MatrixLoader<T>> &&right, uint32_t load_size = 1024)
        : left(std::move(left))
        , right(std::move(right))
        , col_buffer(this->left->rows())
        , val_buffer(load_size)
        , row_buffer(load_size)
        , max_load_size(load_size) {

        if (this->left->rows() != this->right->rows() ||
            this->left->cols() != this->right->cols())
            throw std::runtime_error("Matrices have incompatible dimensions for element-wise addition");
    }

    uint32_t rows() const override { return left->rows(); }
    uint32_t cols() const override { return left->cols(); }

    const char *rowNames(uint32_t row) override { return left->rowNames(row); }
    const char *colNames(uint32_t col) override { return left->colNames(col); }

    // Reset the iterators to start from the beginning
    void restart() override {
        left->restart();
        right->restart();
        row_idx = 0;
    }

    // Seek to a specific column without reading data
    // Next call should be to load(). col must be < cols()
    void seekCol(uint32_t col) override {
        left->seekCol(col);
        right->seekCol(col);
        row_idx = 0;
    }

    // Advance both matrices to the next column,
    // return false if there are no more columns,
    // or error if columns mismatch
    bool nextCol() override {
        bool l = left->nextCol();
        bool r = right->nextCol();
        if (!l && !r) return false;
        if (l != r) throw std::runtime_error("MatrixAddition: left and right nextCol() differs. Matrices desynchronized");
        row_idx = 0;
        return true;
    }
    
    // Return the index of the current column
    uint32_t currentCol() const override { return left->currentCol(); }

    // Return false if there are no more entries to load
    bool load() override {
        if (row_idx == 0) {
            // Load the next column of data into col_buffer
            for (auto &x : col_buffer) {
                x = 0;
            }

            // Add each non-zero entry in matrix column to col_buffer
            accumulate_column(*left);
            accumulate_column(*right);
        }

        loaded = 0;
        for (; row_idx < col_buffer.size() && loaded < max_load_size; row_idx++) {
            if (col_buffer[row_idx] == 0) continue;
            val_buffer[loaded] = col_buffer[row_idx];
            row_buffer[loaded] = row_idx;
            loaded += 1;
        }
        return loaded > 0;
    }

    // Number of loaded entries available
    uint32_t capacity() const override { return loaded; }

    // Pointers to the loaded entries
    uint32_t *rowData() override { return row_buffer.data(); }
    T *valData() override { return val_buffer.data(); }

    // MATH OPERATIONS: utilize distributive property that (A+B)*v = A*v + B*v
    Eigen::VectorXd vecMultiplyRight(
        const Eigen::Map<Eigen::VectorXd> v, std::atomic<bool> *user_interrupt = NULL
    ) override {
        return left->vecMultiplyRight(v, user_interrupt) +
               right->vecMultiplyRight(v, user_interrupt);
    }
    Eigen::VectorXd vecMultiplyLeft(
        const Eigen::Map<Eigen::VectorXd> v, std::atomic<bool> *user_interrupt = NULL
    ) override {
        return left->vecMultiplyLeft(v, user_interrupt) +
               right->vecMultiplyLeft(v, user_interrupt);
    }
    // Linearity of sums (distribution over addition):
    //     rowSums(A+B) = rowSums(A) + rowSums(B) 
    //     colSums(A+B) = colSums(A) + colSums(B) 
    std::vector<T> colSums(std::atomic<bool> *user_interrupt = NULL) override {
        auto l = left->colSums(user_interrupt);
        auto r = right->colSums(user_interrupt);
        for (uint32_t i = 0; i < l.size(); i++) {
            l[i] += r[i];
        }
        return l;
    }
    std::vector<T> rowSums(std::atomic<bool> *user_interrupt = NULL) override {
        auto l = left->rowSums(user_interrupt);
        auto r = right->rowSums(user_interrupt);
        for (uint32_t i = 0; i < l.size(); i++) {
            l[i] += r[i];
        }
        return l;
    }
};

} // end namespace BPCells
