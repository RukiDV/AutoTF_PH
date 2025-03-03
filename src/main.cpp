#include "volume.hpp"
#include "gpu_renderer.hpp"
#include "persistence.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>

int main(int argc, char* argv[])
{
    // get volume from the file provided as cli argument
    Volume volume;

    if (argc > 1) 
    {
        std::string path = argv[1];
        std::cout << "Loading volume from file: " << path << std::endl;

        if (load_volume_from_file(path, volume) != 0) 
        {
            std::cerr << "Failed to parse file!" << std::endl;
        }
    } 
    else 
    {
        std::cout << "No file provided. Using default 2x2x2 volume." << std::endl;
        volume = create_small_volume();
    }

    // Erstelle die Boundary-Matrix und extrahiere die Filtrationswerte
    auto [boundary_matrix, filtration_values] = create_boundary_matrix_from_volume(volume);

    // Sortiere die Filtrationswerte
    std::vector<uint32_t> indices(filtration_values.size());
    for (uint32_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&filtration_values](uint32_t a, uint32_t b) {
        return filtration_values[a] < filtration_values[b];
    });

    // Berechne die Persistenzpaare
    std::vector<PersistencePair> pairs = boundary_matrix.reduce();

    // Drucke die Persistenzpaare
    std::cout << "Persistence Pairs:" << std::endl;
    for (const auto& pair : pairs) {
        std::cout << "(" << pair.birth << ", " << pair.death << ")" << std::endl;
    }

    std::ofstream outfile("persistence_pairs.txt");
    if (!outfile.is_open()) 
    {
        std::cerr << "Failed to open persistence_pairs.txt for writing!" << std::endl;
        return 1;
    }

    for (const auto& [birth, death] : pairs) {
        outfile << birth << " " << death << "\n";
    }
    outfile.close();
    std::cout << "Persistence pairs saved to persistence_pairs.txt" << std::endl; 

    // Visualisiere das Volumen
    //visualize_volume(volume);

    if (gpu_render(volume) != 0) return 1;

    return 0;
}