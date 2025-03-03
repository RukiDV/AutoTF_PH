#pragma once

#include <vector>
#include <utility>
#include <stdint.h>
#include <string>

struct PersistencePair 
{
    uint32_t birth;
    uint32_t death;

    PersistencePair(uint32_t b, uint32_t d) : birth(b), death(d) {}
};

class BoundaryMatrix 
{
public:
    BoundaryMatrix(uint32_t num_cols);

    // sets the dimension of a simplex
    void set_dim(uint32_t col_idx, uint32_t dim);

    // sets a column of the matrix
    void set_col(uint32_t col_idx, const std::vector<uint32_t>& entries);

    // returns the number of columns
    uint32_t get_num_cols() const;

    // returns the entries of a column
    std::vector<uint32_t> get_col(uint32_t col_idx) const;

    // performs reduction and computes persistence pairs
    std::vector<PersistencePair> reduce();

    // function to convert the boundary matrix into a 2D array
    std::vector<std::vector<uint32_t>> matrix_to_2d(const BoundaryMatrix& matrix, uint32_t num_rows); 
   
    // function to print a matrix
    void print_matrix(const std::vector<std::vector<uint32_t>>& matrix, const std::string& name);

    void compute_persistence();

private:
    uint32_t num_cols;
    std::vector<std::vector<uint32_t>> matrix;
    std::vector<uint32_t> dims;

    // add one column to another
    void add_to(uint32_t source_col, uint32_t target_col);
};