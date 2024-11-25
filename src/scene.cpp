#include "scene.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

[[nodiscard]] int load_scene_from_file(const std::string& path, Scene& scene)
{
  std::ifstream file(path);
  if (!file.is_open()) return 1;
  // TODO(Ruki): read in volume file
  scene.resolution.x = 103;
  scene.resolution.y = 94;
  scene.resolution.z = 161;
  
  size_t volume_size = scene.resolution.x * scene.resolution.y * scene.resolution.z;
  scene.data.resize(volume_size);
  file.read(reinterpret_cast<char*>(scene.data.data()), volume_size);
  
  if (!file) 
  {
    std::cerr << "Failed to read file!" << std::endl;
    return 1;
  }

  file.close();
  return 0;
}
