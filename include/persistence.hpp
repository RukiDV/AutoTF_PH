#pragma once

#include <vector>
#include <cstdint>
#include <utility>

struct Volume;

struct PersistencePair 
{
    uint32_t birth = 0;
    uint32_t death = 0;

    PersistencePair() : birth(0), death(0) {}
    PersistencePair(uint32_t b, uint32_t d) : birth(b), death(d) {}

    // calculate persistence
    uint32_t persistence() const { return death - birth; }
};

class BoundaryMatrix 
{
public:
    BoundaryMatrix(uint32_t num_cols);

    void set_dim(uint32_t col_idx, uint32_t dim);
    void set_col(uint32_t col_idx, const std::vector<uint32_t>& entries);
    uint32_t get_num_cols() const;
    std::vector<uint32_t> get_col(uint32_t col_idx) const;

    std::vector<PersistencePair> reduce();

private:
    uint32_t num_cols_;
    std::vector<std::vector<uint32_t>> matrix_;
    std::vector<uint32_t> dims_;

    void add_to(uint32_t source_col, uint32_t target_col);
};

std::pair<BoundaryMatrix, std::vector<int>> create_boundary_matrix(const Volume& volume);
