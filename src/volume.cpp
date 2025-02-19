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
    std::ifstream file(raw_file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) 
    {
        std::cerr << "Failed to open raw data file: " << raw_file_path << std::endl;
        return 1;
    }
    std::cout << "Successfully opened raw data file: " << raw_file_path << std::endl;

    // Bestimme Dateigröße
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize != static_cast<std::streamsize>(volume_size)) {
        std::cerr << "File size mismatch! Read: " << fileSize << ", expected: " << volume_size << std::endl;
        return 1;
    }
    // read volume data
    //file.read(reinterpret_cast<char*>(volume.data.data()), volume_size);
    file.read(reinterpret_cast<char*>(volume.data.data()), volume_size);
    std::cout << "Bytes read: " << file.gcount() << " expected: " << volume_size << std::endl;

    if (!file) 
    {
        std::cerr << "Failed to read raw data file!" << std::endl;
        return 1;
    }

    file.close();

    // Debug: print first few values of the volume data
    std::cout << "Volume data loaded successfully. First few values:" << std::endl;
    std::cout << "First 50 bytes raw from memory:\n";
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


// calculate the index of a voxel in the 3D matrix
uint32_t get_voxel_index(uint32_t x, uint32_t y, uint32_t z, uint32_t dim_x, uint32_t dim_y) 
{
    return z * dim_x * dim_y + y * dim_x + x;
}

bool create_boundary_matrix(const Volume& volume, persistence::BoundaryMatrix& boundary_matrix) 
{
    const size_t dim_x = volume.resolution.x;
    const size_t dim_y = volume.resolution.y;
    const size_t dim_z = volume.resolution.z;

    auto get_voxel_index = [dim_x, dim_y](size_t x, size_t y, size_t z) 
    {
        return z * dim_x * dim_y + y * dim_x + x;
    };

    size_t num_entries = 0;

    for (size_t z = 0; z < dim_z; ++z) 
    {
        for (size_t y = 0; y < dim_y; ++y) 
        {
            for (size_t x = 0; x < dim_x; ++x) 
            {
                size_t voxel_index = get_voxel_index(x, y, z);
                uint8_t voxel_value = volume.data[voxel_index];

                if (voxel_value == 0) continue;  // ignore background

                persistence::column column_entries;

                // add edge for neighboring voxel
                if (x > 0) 
                {
                    size_t neighbor_index = get_voxel_index(x - 1, y, z);
                    if (volume.data[neighbor_index] <= voxel_value) 
                    {
                        column_entries.push_back(neighbor_index);
                    }
                }
                if (y > 0) 
                {
                    size_t neighbor_index = get_voxel_index(x, y - 1, z);
                    if (volume.data[neighbor_index] <= voxel_value) 
                    {
                        column_entries.push_back(neighbor_index);
                    }
                }
                if (z > 0) 
                {
                    size_t neighbor_index = get_voxel_index(x, y, z - 1);
                    if (volume.data[neighbor_index] <= voxel_value) 
                    {
                        column_entries.push_back(neighbor_index);
                    }
                }
                // save column in boundary matrix
                if (!column_entries.empty()) {
                    boundary_matrix.set_col(voxel_index, column_entries);
                    num_entries += column_entries.size();
                    //std::cout << "Column " << voxel_index << " has " << column_entries.size() << " entries.\n";
                }
            }
        }
    }
    std::cout << "Boundary Matrix constructed with " << num_entries << " nonzero entries.\n";
    
    return true;
}




