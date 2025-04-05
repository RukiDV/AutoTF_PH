#pragma once
#include <string>
#include <vector>
#include "glm/vec3.hpp"
#include <stdexcept>

enum class FiltrationMode {
    LowerStar, // use the maximum
    UpperStar // use the minimum
};

struct Volume
{
  glm::uvec3 resolution;
  std::vector<uint8_t> data;
};

[[nodiscard]] int load_volume_from_file(const std::string& path, Volume& volume);