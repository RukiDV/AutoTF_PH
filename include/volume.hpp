#pragma once
#include <string>
#include <vector>
#include "glm/vec3.hpp"
#include <stdexcept>

enum class FiltrationMode
{
    LowerStar, // use the maximum
    UpperStar // use the minimum
};

struct Volume
{
  std::string name;
  glm::uvec3 resolution;
  std::vector<uint8_t> data;
};

[[nodiscard]] int load_volume_from_file(const std::string& path, Volume& volume);
Volume compute_gradient_volume(const Volume& volume);
Volume create_simple_volume();
Volume create_disjoint_components_volume();
Volume create_tiny_disjoint_volume();
Volume create_gradient_volume();