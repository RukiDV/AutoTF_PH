#include "volume.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

[[nodiscard]] int load_volume_from_file(const std::string& path, Volume& volume)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return 1;
    }

    // resolution of volume
    volume.resolution.x = 103;
    volume.resolution.y = 94;
    volume.resolution.z = 161;

    // calculate expected size of volume
    size_t volume_size = volume.resolution.x * volume.resolution.y * volume.resolution.z;
    volume.data.resize(volume_size);

    // check size of file
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size != volume_size) {
        std::cerr << "File size does not match expected volume size!" << std::endl;
        std::cerr << "Expected: " << volume_size << " bytes, but got: " << file_size << " bytes." << std::endl;
        return 1;
    }

    // read volume data
    file.read(reinterpret_cast<char*>(volume.data.data()), volume_size);

    if (!file) {
        std::cerr << "Failed to read file!" << std::endl;
        return 1;
    }

    file.close();

    // Debug: print first few values to verify
    std::cout << "Volume data loaded successfully. First few values:" << std::endl;
    for (size_t i = 0; i < 10 && i < volume_size; ++i) {
        std::cout << static_cast<int>(volume.data[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}
