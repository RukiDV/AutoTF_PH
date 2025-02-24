#include "simple_per_homo_example.hpp"
#include <iostream>

// function to convert the boundary matrix into a 2D array
std::vector<std::vector<uint32_t>> matrix_to_2d(const BoundaryMatrix& matrix, uint32_t num_rows) 
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

// function to print a matrix
void print_matrix(const std::vector<std::vector<uint32_t>>& matrix, const std::string& name) 
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

int main() 
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
    std::vector<std::vector<uint32_t>> original_matrix = matrix_to_2d(boundary_matrix, num_rows);

    print_matrix(original_matrix, "Original Boundary-Matrix");

    // perform reduction and compute persistence pairs
    std::vector<PersistencePair> pairs = boundary_matrix.reduce();

    // convert the reduced boundary matrix into a 2D array
    std::vector<std::vector<uint32_t>> reduced_matrix = matrix_to_2d(boundary_matrix, num_rows);

    print_matrix(reduced_matrix, "Reduced Boundary-Matrix");

    std::cout << "\nPersistence Pairs:" << std::endl;
    for (const auto& pair : pairs) 
    {
        std::cout << "(" << pair.birth << ", " << pair.death << ")" << std::endl;
    }

    return 0;
}