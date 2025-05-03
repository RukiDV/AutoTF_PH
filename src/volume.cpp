#include "volume.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include <vector>
#include <cstdint>
#include <cmath>

[[nodiscard]] int load_volume_from_file(const std::string& header_filename, Volume& volume)
{
    const std::string volume_folder = "data/volume/";
    if (!std::filesystem::exists(volume_folder)) std::filesystem::create_directories(volume_folder);
    volume.name = header_filename.substr(0, header_filename.find_last_of('.'));
    const std::string header_path = volume_folder + header_filename;
    std::ifstream header_file(header_path);
    if (!header_file.is_open()) {
        std::cerr << "Failed to open header file: " << header_path << std::endl;
        return 1;
    }

    std::string line;
    bool sizes_found = false;
    std::string data_file_path;

    // iterate through each line of header file
    while (std::getline(header_file, line)) 
    {
        // look for 'sizes:' line to extract resolution
        if (line.find("sizes:") == 0) 
        {
            std::stringstream ss(line.substr(6));  // skip "sizes:"
            ss >> volume.resolution.x >> volume.resolution.y >> volume.resolution.z;
            std::cout << "Parsed sizes: " << volume.resolution.x << " " << volume.resolution.y << " " << volume.resolution.z << std::endl;
            sizes_found = true;
        }
        // look for 'data file:' line to extract path to the .raw file
        if (line.find("data file:") == 0) 
        {
            data_file_path = line.substr(10); // skip "data file:"
            // remove whitespaces
            data_file_path.erase(data_file_path.begin(), std::find_if(data_file_path.begin(), data_file_path.end(), [](unsigned char ch) 
            {
                return !std::isspace(ch);
            }));
            data_file_path.erase(std::find_if(data_file_path.rbegin(), data_file_path.rend(), [](unsigned char ch) 
            {
                return !std::isspace(ch);
            }).base(), data_file_path.end());
            std::cout << "Data file path: " << data_file_path << std::endl;
        }
    }

    header_file.close();

    // check if resolution and path to the .raw file were correctly read
    if (!sizes_found || volume.resolution.x == 0 || volume.resolution.y == 0 || volume.resolution.z == 0) 
    {
        std::cerr << "Failed to parse volume resolution!" << std::endl;
        return 1;
    }
    if (data_file_path.empty()) 
    {
        std::cerr << "Failed to parse data file path from header!" << std::endl;
        return 1;
    }

    // calculate expected size of volume
    size_t volume_size = volume.resolution.x * volume.resolution.y * volume.resolution.z;
    volume.data.resize(volume_size);

    // combine directory path of header file with the path to the .raw file
    std::filesystem::path raw_file_path = std::filesystem::path(header_path).parent_path() / data_file_path;

    // open volume file (raw file)
    std::ifstream file(raw_file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) 
    {
        std::cerr << "Failed to open raw data file: " << raw_file_path << std::endl;
        return 1;
    }
    std::cout << "Successfully opened raw data file: " << raw_file_path << std::endl;

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize != static_cast<std::streamsize>(volume_size)) {
        std::cerr << "File size mismatch! Read: " << fileSize << ", expected: " << volume_size << std::endl;
        return 1;
    }
    // read volume data
    file.read(reinterpret_cast<char*>(volume.data.data()), volume_size);
    std::cout << "Bytes read: " << file.gcount() << " expected: " << volume_size << std::endl;

    if (!file) 
    {
        std::cerr << "Failed to read raw data file!" << std::endl;
        return 1;
    }

    file.close();

    // Debug: print first few values of the volume data
    std::cout << "Volume data loaded successfully." << std::endl;
    float avg = 0.0f;
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();
    float variance = 0.0f;
    float std = 0.0f;
    for (const uint8_t value : volume.data) 
    {
        avg += float(value) / float(volume.data.size());
        min = std::min(min, float(value));
        max = std::max(max, float(value));
    }
    std::printf("avg: %.3f, min: %.3f, max: %.3f", avg, min, max);
    std::cout << std::endl;

    size_t num_non_zero = 0;
    for (size_t i = 0; i < volume.data.size(); ++i) 
    {
        if (volume.data[i] != 0) 
        {
            num_non_zero++;
        }
    }
    std::cout << "Number of non-zero voxels: " << num_non_zero << std::endl;

    return 0;
}

Volume create_simple_volume()
{
    Volume volume;
    volume.resolution = glm::uvec3(16, 16, 16);
    uint32_t totalVoxels = 16 * 16 * 16;
    // big outer cube
    volume.data.assign(totalVoxels, 128u);

    auto idx = [&](uint32_t x, uint32_t y, uint32_t z)
    {
        return size_t(z) * 16 * 16 + size_t(y) * 16 + size_t(x);
    };

    // middle cube
    for (uint32_t z = 4; z < 12; ++z)
        for (uint32_t y = 4; y < 12; ++y)
            for (uint32_t x = 4; x < 12; ++x)
                volume.data[idx(x,y,z)] = 230u;

    // small cube
    for (uint32_t z = 6; z < 10; ++z)
        for (uint32_t y = 6; y < 10; ++y)
            for (uint32_t x = 6; x < 10; ++x)
                volume.data[idx(x,y,z)] = 255u;

    return volume;
}

Volume create_gradient_volume()
{
    Volume volume;
    volume.resolution = glm::uvec3(16, 16, 16);
    uint32_t totalVoxels = 16 * 16 * 16;
    volume.data.resize(totalVoxels);
    
    // Create a gradient in the x-direction.
    // Voxels at x=0 will have intensity 0 and at x=15 intensity 255.
    for (uint32_t z = 0; z < 16; ++z) {
        for (uint32_t y = 0; y < 16; ++y) {
            for (uint32_t x = 0; x < 16; ++x) {
                uint8_t intensity = static_cast<uint8_t>((x / 15.0f) * 255);
                volume.data[z * 16 * 16 + y * 16 + x] = intensity;
            }
        }
    }
    return volume;
}