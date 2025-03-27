#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "volume.hpp"
#include "persistence.hpp"
#include "merge_tree.hpp"
#include "app_state.hpp"
#include "gpu_renderer.hpp"

int main(int argc, char* argv[])
{
    Volume volume;
    if (argc > 1) 
    {
        std::string path = argv[1];
        std::cout << "Loading volume from file: " << path << std::endl;
        if (load_volume_from_file(path, volume) != 0) 
        {
            std::cerr << "Failed to load volume!" << std::endl;
            return 1;
        }
    } else 
    {
        std::cout << "No file provided. Using default small volume." << std::endl;
        volume = create_small_volume();
    }

    if (gpu_render(volume) != 0) 
    {
        std::cerr << "Failed to render volume on GPU!" << std::endl;
        return 1;
    }
    return 0;
}