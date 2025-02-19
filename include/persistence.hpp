#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace persistence 
{
    using index = int; 
    using column = std::vector<index>;

    class BoundaryMatrix 
    {
    public:
        BoundaryMatrix(size_t rows, size_t cols);

        index get_num_cols() const { return cols; }
        index get_max_index(index col) const;
        void add_to(index source_col, index target_col);
        void finalize(index col);

        // additional methods for phat compatibility
        void set_col(index idx, const column& col);
        const column& get_col(index idx) const;
        void clear(index idx);

    private:
        std::vector<column> matrix;
        size_t rows, cols;
    };

    class ExtendedReduction 
    {
    public:
        void operator()(BoundaryMatrix& boundary_matrix, std::unordered_map<index, index>& absorptions, std::unordered_map<index, std::unordered_set<index>>& connected_components);
    }; 

} // namespace persistence