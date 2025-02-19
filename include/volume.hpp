#pragma once
#include <string>
#include <vector>
#include "glm/vec3.hpp"
#include "persistence.hpp"

struct Volume
{
  glm::uvec3 resolution;
  std::vector<uint8_t> data;
};

[[nodiscard]] int load_volume_from_file(const std::string& path, Volume& volume);
bool create_boundary_matrix(const Volume& volume, persistence::BoundaryMatrix& boundary_matrix);
