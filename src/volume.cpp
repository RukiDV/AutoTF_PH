#include "volume.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

[[nodiscard]] int load_volume_from_file(const std::string& header_path, Volume& volume)
{
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
    std::ifstream file(raw_file_path, std::ios::binary);
    if (!file.is_open()) 
    {
        std::cerr << "Failed to open raw data file: " << raw_file_path << std::endl;
        return 1;
    }

    // read volume data
    file.read(reinterpret_cast<char*>(volume.data.data()), volume_size);

    if (!file) 
    {
        std::cerr << "Failed to read raw data file!" << std::endl;
        return 1;
    }

    file.close();

    // Debug: print first few values of the volume data
    std::cout << "Volume data loaded successfully. First few values:" << std::endl;
    for (size_t i = 0; i < 10 && i < volume_size; ++i) {
        std::cout << static_cast<int>(volume.data[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}