#include "simple_per_homo_example.hpp"
#include <iostream>
#include <algorithm>

BoundaryMatrix::BoundaryMatrix(uint32_t num_cols) : num_cols(num_cols), matrix(num_cols, std::vector<uint32_t>()), dims(num_cols, 0) {}

void BoundaryMatrix::set_dim(uint32_t col_idx, uint32_t dim) 
{
    if (col_idx >= 0 && col_idx < num_cols) 
    {
        dims[col_idx] = dim;
    }
}

void BoundaryMatrix::set_col(uint32_t col_idx, const std::vector<uint32_t>& entries) 
{
    if (col_idx >= 0 && col_idx < num_cols) 
    {
        matrix[col_idx] = entries;
    }
}

uint32_t BoundaryMatrix::get_num_cols() const 
{
    return num_cols;
}

std::vector<uint32_t> BoundaryMatrix::get_col(uint32_t col_idx) const 
{
    if (col_idx >= 0 && col_idx < num_cols) 
    {
        return matrix[col_idx];
    }
    return {};
}

// performs reduction and computes persistence pairs using the standard reduction algorithm
std::vector<PersistencePair> BoundaryMatrix::reduce() 
{
    std::vector<PersistencePair> pairs;
    std::vector<uint32_t> lowest_one_lookup(num_cols, -1); // Tracks the pivot elements

    for (uint32_t cur_col = 0; cur_col < num_cols; ++cur_col) 
    {
        if (!matrix[cur_col].empty()) 
        {
            // find the highest entry (pivot) in the current column
            uint32_t lowest_one = *std::max_element(matrix[cur_col].begin(), matrix[cur_col].end());

            // while the current pivot is already used as a pivot in another column
            while (lowest_one != -1 && lowest_one_lookup[lowest_one] != -1) 
            {
                // Subtract the corresponding column
                add_to(lowest_one_lookup[lowest_one], cur_col);
                if (!matrix[cur_col].empty()) 
                {
                    lowest_one = *std::max_element(matrix[cur_col].begin(), matrix[cur_col].end());
                } else 
                {
                    lowest_one = -1;
                }
            }

            // if a new pivot is found, store it
            if (lowest_one != -1) 
            {
                lowest_one_lookup[lowest_one] = cur_col;
                pairs.emplace_back(lowest_one, cur_col);
            }
        }
    }
    return pairs;
}

// helper function to add one column to another (XOR operation)
void BoundaryMatrix::add_to(uint32_t source_col, uint32_t target_col) 
{
    for (uint32_t entry : matrix[source_col]) 
    {
        auto it = std::find(matrix[target_col].begin(), matrix[target_col].end(), entry);
        if (it != matrix[target_col].end()) 
        {
            matrix[target_col].erase(it); // remove the entry if it exists
        } else 
        {
            matrix[target_col].push_back(entry); // add the entry if it doesn't exist
        }
    }
}
// function to convert the boundary matrix into a 2D array
    std::vector<std::vector<uint32_t>> BoundaryMatrix::matrix_to_2d(const BoundaryMatrix& matrix, uint32_t num_rows) 
    {
        uint32_t num_cols = matrix.get_num_cols();
        std::vector<std::vector<uint32_t>> result(num_rows, std::vector<uint32_t>(num_cols, 0));
        for (uint32_t col_idx = 0; col_idx < num_cols; ++col_idx) 
        {
            std::vector<uint32_t> temp_col = matrix.get_col(col_idx);
            for (uint32_t row_idx : temp_col) 
            {
                if (row_idx < num_rows) 
                {
                    result[row_idx][col_idx] = 1;
                }
            }
        }
        return result;
    }

void BoundaryMatrix::print_matrix(const std::vector<std::vector<uint32_t>>& matrix, const std::string& name) 
{
    std::cout << name << ":" << std::endl;
    for (const auto& row : matrix) 
    {
        for (uint32_t val : row) 
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void BoundaryMatrix::compute_persistence()
{
    BoundaryMatrix boundary_matrix(12); // 12 simplices

    // Set the dimensions of the simplices
    boundary_matrix.set_dim(0, 0); // point
    boundary_matrix.set_dim(1, 0);
    boundary_matrix.set_dim(2, 1); // edge
    boundary_matrix.set_dim(3, 0); 
    boundary_matrix.set_dim(4, 1);
    boundary_matrix.set_dim(5, 1);
    boundary_matrix.set_dim(6, 0);
    boundary_matrix.set_dim(7, 0);
    boundary_matrix.set_dim(8, 1);
    boundary_matrix.set_dim(9, 1);
    boundary_matrix.set_dim(10, 1);
    boundary_matrix.set_dim(11, 2); // triangle

    // fill the columns of the boundary matrix
    boundary_matrix.set_col(2, {0, 1}); // edge (0, 1)
    boundary_matrix.set_col(4, {0, 3}); // edge (0, 3)
    boundary_matrix.set_col(5, {1, 3}); // edge (1, 3)
    boundary_matrix.set_col(8, {6, 7}); // edge (6, 7)
    boundary_matrix.set_col(9, {3, 7}); // edge (3, 7)
    boundary_matrix.set_col(10, {1, 6}); // edge (1, 6)
    boundary_matrix.set_col(11, {2, 4, 5}); // triangle (2, 4, 5)

    // Determine the number of rows for the matrix representation
    uint32_t num_rows = 12;

    // convert the boundary matrix into a 2D array
    std::vector<std::vector<uint32_t>> original_matrix =boundary_matrix.matrix_to_2d(boundary_matrix, num_rows);

    boundary_matrix.print_matrix(original_matrix, "Original Boundary-Matrix");

    // perform reduction and compute persistence pairs
    std::vector<PersistencePair> pairs = boundary_matrix.reduce();

    // convert the reduced boundary matrix into a 2D array
    std::vector<std::vector<uint32_t>> reduced_matrix = boundary_matrix.matrix_to_2d(boundary_matrix, num_rows);

    std::cout << std::endl;
    boundary_matrix.print_matrix(reduced_matrix, "Reduced Boundary-Matrix");

    std::cout << "\nPersistence Pairs:" << std::endl;
    for (const auto& pair : pairs) 
    {
        std::cout << "(" << pair.birth << ", " << pair.death << ")" << std::endl;
    }
}

