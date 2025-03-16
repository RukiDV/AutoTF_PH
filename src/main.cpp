#include "volume.hpp"
#include "gpu_renderer.hpp"
#include "persistence.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>

int main(int argc, char* argv[])
{
    Volume volume;

    if (argc > 1) 
    {
        std::string path = argv[1];
        std::cout << "Loading volume from file: " << path << std::endl;
        if (load_volume_from_file(path, volume) != 0) 
        {
            std::cerr << "Failed to parse file!" << std::endl;
            return 1;
        }
    } else 
    {
        std::cout << "No file provided. Using default small volume." << std::endl;
        volume = create_small_volume();
    }

    auto [boundary_matrix, filtration_values] = create_boundary_matrix(volume);

    std::vector<PersistencePair> pairs = boundary_matrix.reduce();

    std::ofstream outfile("persistence_pairs.txt");
    if (!outfile.is_open()) 
    {
        std::cerr << "Failed to open persistence_pairs.txt for writing!" << std::endl;
        return 1;
    }
    for (const auto& [birth, death] : pairs)
    {
        outfile << filtration_values[birth] << " " << (death == -1 ? std::numeric_limits<int>::max() : filtration_values[death]) << "\n";
    }
    outfile.close();
    std::cout << "Persistence pairs saved to persistence_pairs.txt" << std::endl;

    if (gpu_render(volume, pairs) != 0) 
    {
        std::cerr << "Failed to render volume on GPU!" << std::endl;
        return 1;
    }
    return 0;
}