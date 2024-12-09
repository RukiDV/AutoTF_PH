#include <iostream>
#include "gpu_renderer.hpp"
#include "volume.hpp"

int main(int argc, char* argv[])
{
  // get volume from the file provided as cli argument
  std::string path;
  if (argc > 1) path = argv[1];
  else
  {
    std::cout << "Provide path of file to load!" << std::endl;
    return 1;
  }
  Volume volume;
  if (load_volume_from_file(path, volume) != 0)
  {
    std::cerr << "Failed to parse file!" << std::endl;
    return 1;
  }

  AppState app_state;
  GPU_Renderer renderer(app_state, volume); 
  if (renderer.gpu_render(volume) != 0) return 1;
  return 0;
}
