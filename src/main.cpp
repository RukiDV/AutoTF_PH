#include <iostream>
#include <fstream>

#include "gpu_renderer.hpp"
#include "persistence.hpp"
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

    uint32_t num_voxels = volume.resolution.x * volume.resolution.y * volume.resolution.z;
    persistence::BoundaryMatrix boundary_matrix(num_voxels, num_voxels);

    // fill boundary matrix
    if (!create_boundary_matrix(volume, boundary_matrix)) {
        std::cerr << "Failed to create boundary matrix!" << std::endl;
        return 1;
    }

    // calculate persistente homologie
    persistence::ExtendedReduction reduction;
    std::unordered_map<persistence::index, persistence::index> absorptions;
    std::unordered_map<persistence::index, std::unordered_set<persistence::index>> connected_components;

    reduction(boundary_matrix, absorptions, connected_components);

    std::cout << "Number of persistence pairs: " << absorptions.size() << std::endl;
    for (const auto& [birth, death] : absorptions) {
        std::cout << "Persistence pair: (" << birth << ", " << death << ")" << std::endl;
    }

    std::ofstream outfile("persistence_pairs.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open persistence_pairs.txt for writing!" << std::endl;
        return 1;
    }

    for (const auto& [birth, death] : absorptions) {
        outfile << birth << " " << death << "\n";
    }
    outfile.close();

    std::cout << "Persistence pairs saved to persistence_pairs.txt" << std::endl; 

    if (gpu_render(volume) != 0) return 1;

    return 0;
}