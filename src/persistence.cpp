#include "persistence.hpp"

#include <algorithm>
#include <iostream> 
#include <cstdint>

namespace persistence 
{
    BoundaryMatrix::BoundaryMatrix(size_t rows, size_t cols) : rows(rows), cols(cols), matrix(cols, column()) {}

    // returns index of the lowest one in a column
    index BoundaryMatrix::get_max_index(index col) const 
    {
        if (matrix[col].empty()) 
        {
            //std::cout << "Column " << col << " is empty, returning -1.\n";
            return -1;
        }
        //std::cout << "Column " << col << " has " << matrix[col].size() << " entries. Max index: " << matrix[col].back() << "\n";
        return matrix[col].back();
    }

    // adds column source_col to column target_col (mod 2)
    void BoundaryMatrix::add_to(index source_col, index target_col) 
    {
        const column& source = matrix[source_col];
        column& target = matrix[target_col];

        if (source.empty()) 
        {
            std::cout << "Source column " << source_col << " is empty, nothing to add.\n";
            return;
        }

        /*std::cout << "Before sorting - Source column " << source_col << ": ";
        for (auto val : source) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Before sorting - Target column " << target_col << ": ";
        for (auto val : target) std::cout << val << " ";
        std::cout << "\n";*/

        // Ensure the columns are sorted
        column sorted_source = source;
        column sorted_target = target;
        std::sort(sorted_source.begin(), sorted_source.end());
        std::sort(sorted_target.begin(), sorted_target.end());

        /*std::cout << "After sorting - Source column " << source_col << ": ";
        for (auto val : sorted_source) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "After sorting - Target column " << target_col << ": ";
        for (auto val : sorted_target) std::cout << val << " ";
        std::cout << "\n";*/

        column temp_col;
        std::set_symmetric_difference(sorted_target.begin(), sorted_target.end(), sorted_source.begin(), sorted_source.end(), std::back_inserter(temp_col));

        /*std::cout << "Temporary column: ";
        for (auto val : temp_col) std::cout << val << " ";
        std::cout << "\n";*/

        // Assign the result to the target column
        target.swap(temp_col);
        finalize(target_col);

        // Debug: Print the target column after finalization
        /*std::cout << "After finalization - Target column " << target_col << ": ";
        for (auto val : target) std::cout << val << " ";
        std::cout << "\n";*/
    }
 
    // finalize column, sort and remove duplicates
    void BoundaryMatrix::finalize(index col) 
    {
        if (matrix[col].empty()) return;

        std::sort(matrix[col].begin(), matrix[col].end());
        matrix[col].erase(std::unique(matrix[col].begin(), matrix[col].end()), matrix[col].end());
    }

    void BoundaryMatrix::set_col(index idx, const column& col) 
    {
        matrix[idx] = col;
    }

    const column& BoundaryMatrix::get_col(index idx) const 
    {
        return matrix[idx];
    }

    void BoundaryMatrix::clear(index idx) 
    {
        matrix[idx].clear();
    }

    // standard reduction algorithm with absorption and connected components tracking
    void ExtendedReduction::operator()(BoundaryMatrix& boundary_matrix, std::unordered_map<index, index>& absorptions, std::unordered_map<index, std::unordered_set<index>>& connected_components) {
        const index nr_columns = boundary_matrix.get_num_cols();
        std::vector<index> lowest_one_lookup(nr_columns, -1);

        // initialize connected components, each column starts as its own component
        for (index i = 0; i < nr_columns; ++i) 
        {
            connected_components[i].insert(i); // each column contains only itself
        }

        for (index cur_col = 0; cur_col < nr_columns; ++cur_col) 
        {
        index lowest_one = boundary_matrix.get_max_index(cur_col);

        if (lowest_one != -1) 
        {
            lowest_one_lookup[lowest_one] = cur_col;
            std::cout << "Column " << cur_col << " is now the lowest_one for index " << lowest_one << "\n";
        }

        while (lowest_one != -1 && lowest_one_lookup[lowest_one] != -1) 
        {
            index absorbing_col = lowest_one_lookup[lowest_one];

            // record absorption (current column is absorbed by absorbing_col)
            absorptions[cur_col] = absorbing_col;

            // merge connected components
            connected_components[absorbing_col].insert(connected_components[cur_col].begin(), connected_components[cur_col].end());
            connected_components[cur_col].clear(); // clear cur_col as it is absorbed

            // reduce current column
            boundary_matrix.add_to(absorbing_col, cur_col);
            lowest_one = boundary_matrix.get_max_index(cur_col);
        }

        if (lowest_one != -1) 
        {
            lowest_one_lookup[lowest_one] = cur_col;
        }

        boundary_matrix.finalize(cur_col);
    }
        std::cout << "Number of persistence pairs: " << absorptions.size() << std::endl;
    }
} // namespace persistence