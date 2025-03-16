#include "persistence.hpp"
#include <iostream>
#include <algorithm>
#include "volume.hpp"

BoundaryMatrix::BoundaryMatrix(uint32_t num_cols) : num_cols_(num_cols), matrix_(num_cols, std::vector<uint32_t>()), dims_(num_cols, 0) {}

// set the dimension of a simplex
void BoundaryMatrix::set_dim(uint32_t col_idx, uint32_t dim) 
{
    if (col_idx < num_cols_) 
    {
        dims_[col_idx] = dim;
    }
}

// set a column in the matrix
void BoundaryMatrix::set_col(uint32_t col_idx, const std::vector<uint32_t>& entries) 
{
    if (col_idx < num_cols_) 
    {
        matrix_[col_idx] = entries;
    }
}

// return the number of columns
uint32_t BoundaryMatrix::get_num_cols() const 
{
    return num_cols_;
}

// return the entries of a column
std::vector<uint32_t> BoundaryMatrix::get_col(uint32_t col_idx) const 
{
    if (col_idx < num_cols_) 
    {
        return matrix_[col_idx];
    }
    return {};
}

// perform the reduction
std::vector<PersistencePair> BoundaryMatrix::reduce() 
{
    std::vector<PersistencePair> pairs;
    std::vector<int> lowest_one_lookup(num_cols_, -1);

    for (uint32_t cur_col = 0; cur_col < num_cols_; ++cur_col) 
    {
        if (!matrix_[cur_col].empty()) 
        {
            uint32_t lowest_one = *std::max_element(matrix_[cur_col].begin(), matrix_[cur_col].end());

            while (lowest_one != 0 && lowest_one_lookup[lowest_one] != -1) 
            {
                add_to(lowest_one_lookup[lowest_one], cur_col);
                if (!matrix_[cur_col].empty()) 
                {
                    lowest_one = *std::max_element(matrix_[cur_col].begin(), matrix_[cur_col].end());
                } else 
                {
                    lowest_one = 0;
                }
            }

            if (lowest_one != 0) 
            {
                lowest_one_lookup[lowest_one] = cur_col;
                pairs.emplace_back(lowest_one, cur_col);
            }
        }
    }
    return pairs;
}

// add entries from one column to another
void BoundaryMatrix::add_to(uint32_t source_col, uint32_t target_col) 
{
    for (uint32_t entry : matrix_[source_col]) 
    {
        auto it = std::find(matrix_[target_col].begin(), matrix_[target_col].end(), entry);
        if (it != matrix_[target_col].end()) 
        {
            matrix_[target_col].erase(it);
        } else 
        {
            matrix_[target_col].push_back(entry);
        }
    }
}

// create the boundary matrix from the volume
std::pair<BoundaryMatrix, std::vector<int>> create_boundary_matrix(const Volume& volume) 
{
    const uint32_t dim_x = volume.resolution.x;
    const uint32_t dim_y = volume.resolution.y;
    const uint32_t dim_z = volume.resolution.z;

    uint32_t num_points = dim_x * dim_y * dim_z;
    uint32_t num_edges = 0;
    uint32_t num_faces = 0;
    uint32_t num_voxels = 0;

    std::vector<std::vector<uint32_t>> edges;
    std::vector<std::vector<uint32_t>> faces;
    std::vector<std::vector<uint32_t>> voxels;
    std::vector<int> filtration_values;

    auto index = [&](uint32_t x, uint32_t y, uint32_t z) 
    {
        return z * dim_y * dim_x + y * dim_x + x;
    };

    // add points
    for (uint32_t z = 0; z < dim_z; ++z) 
    {
        for (uint32_t y = 0; y < dim_y; ++y) 
        {
            for (uint32_t x = 0; x < dim_x; ++x) 
            {
                uint32_t voxel_idx = index(x, y, z);
                filtration_values.push_back(volume.data[voxel_idx]);
            }
        }
    }

    // add edges
    for (uint32_t z = 0; z < dim_z; ++z) 
    {
        for (uint32_t y = 0; y < dim_y; ++y) 
        {
            for (uint32_t x = 0; x < dim_x; ++x) 
            {
                uint32_t voxel_idx = index(x, y, z);

                if (x < dim_x - 1) 
                {
                    uint32_t neighbor = index(x + 1, y, z);
                    edges.push_back({voxel_idx, neighbor});
                    filtration_values.push_back(std::max(volume.data[voxel_idx], volume.data[neighbor]));
                    num_edges++;
                }
                if (y < dim_y - 1) 
                {
                    uint32_t neighbor = index(x, y + 1, z);
                    edges.push_back({voxel_idx, neighbor});
                    filtration_values.push_back(std::max(volume.data[voxel_idx], volume.data[neighbor]));
                    num_edges++;
                }
                if (z < dim_z - 1) 
                {
                    uint32_t neighbor = index(x, y, z + 1);
                    edges.push_back({voxel_idx, neighbor});
                    filtration_values.push_back(std::max(volume.data[voxel_idx], volume.data[neighbor]));
                    num_edges++;
                }
            }
        }
    }

    // add faces
    for (uint32_t z = 0; z < dim_z - 1; ++z) 
    {
        for (uint32_t y = 0; y < dim_y - 1; ++y) 
        {
            for (uint32_t x = 0; x < dim_x - 1; ++x) 
            {
                uint32_t voxel_idx = index(x, y, z);

                // face in the x-y plane
                faces.push_back({voxel_idx, index(x + 1, y, z), index(x, y + 1, z), index(x + 1, y + 1, z)});
                filtration_values.push_back(std::max({volume.data[voxel_idx], volume.data[index(x + 1, y, z)], volume.data[index(x, y + 1, z)], volume.data[index(x + 1, y + 1, z)]}));
                num_faces++;

                // face in the y-z plane
                faces.push_back({voxel_idx, index(x, y + 1, z), index(x, y, z + 1), index(x, y + 1, z + 1)});
                filtration_values.push_back(std::max({volume.data[voxel_idx], volume.data[index(x, y + 1, z)], volume.data[index(x, y, z + 1)], volume.data[index(x, y + 1, z + 1)]}));
                num_faces++;

                // face in the x-z plane
                faces.push_back({voxel_idx, index(x + 1, y, z), index(x, y, z + 1), index(x + 1, y, z + 1)});
                filtration_values.push_back(std::max({volume.data[voxel_idx], volume.data[index(x + 1, y, z)], volume.data[index(x, y, z + 1)], volume.data[index(x + 1, y, z + 1)]}));
                num_faces++;
            }
        }
    }

    // add voxels
    for (uint32_t z = 0; z < dim_z - 1; ++z) 
    {
        for (uint32_t y = 0; y < dim_y - 1; ++y) 
        {
            for (uint32_t x = 0; x < dim_x - 1; ++x) 
            {
                uint32_t voxel_idx = index(x, y, z);
                voxels.push_back({voxel_idx, index(x + 1, y, z), index(x, y + 1, z), index(x + 1, y + 1, z), index(x, y, z + 1), index(x + 1, y, z + 1), index(x, y + 1, z + 1), index(x + 1, y + 1, z + 1)});
                filtration_values.push_back(std::max({volume.data[voxel_idx], volume.data[index(x + 1, y, z)], volume.data[index(x, y + 1, z)], volume.data[index(x + 1, y + 1, z)], volume.data[index(x, y, z + 1)], volume.data[index(x + 1, y, z + 1)], volume.data[index(x, y + 1, z + 1)], volume.data[index(x + 1, y + 1, z + 1)]}));
                num_voxels++;
            }
        }
    }

    BoundaryMatrix boundary_matrix(num_points + num_edges + num_faces + num_voxels);
    for (uint32_t i = 0; i < num_points; ++i) boundary_matrix.set_dim(i, 0);
    for (uint32_t i = 0; i < num_edges; ++i) boundary_matrix.set_dim(num_points + i, 1);
    for (uint32_t i = 0; i < num_faces; ++i) boundary_matrix.set_dim(num_points + num_edges + i, 2);
    for (uint32_t i = 0; i < num_voxels; ++i) boundary_matrix.set_dim(num_points + num_edges + num_faces + i, 3);

    for (uint32_t i = 0; i < num_edges; ++i) boundary_matrix.set_col(num_points + i, edges[i]);
    for (uint32_t i = 0; i < num_faces; ++i) boundary_matrix.set_col(num_points + num_edges + i, faces[i]);
    for (uint32_t i = 0; i < num_voxels; ++i) boundary_matrix.set_col(num_points + num_edges + num_faces + i, voxels[i]);

    return {boundary_matrix, filtration_values};
}
