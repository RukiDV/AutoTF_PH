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