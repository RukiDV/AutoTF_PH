#pragma once
#include <string>
#include <vector>
#include "glm/vec3.hpp"

struct Scene
{
  glm::uvec3 resolution;
  std::vector<uint8_t> data;
};

[[nodiscard]] int load_scene_from_file(const std::string& path, Scene& scene);
