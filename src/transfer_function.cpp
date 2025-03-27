#include "transfer_function.hpp"
#include "glm/vec3.hpp"
#include <iostream>
#include <algorithm>
#include <limits>

std::pair<uint32_t, uint32_t> compute_min_max_scalar(const Volume& volume) 
{
    uint32_t min_value = std::numeric_limits<uint32_t>::max();
    uint32_t max_value = std::numeric_limits<uint32_t>::min();
    for (uint32_t v : volume.data)
    {
        min_value = std::min(min_value, v);
        max_value = std::max(max_value, v);
    }
    return {min_value, max_value};
}

void TransferFunction::update(const std::vector<PersistencePair>& pairs, const Volume& volume, std::vector<glm::vec4>& tf_data) 
{
    auto [vol_min, vol_max] = compute_min_max_scalar(volume);
    std::cout << "Volume scalar range: " << vol_min << " to " << vol_max << std::endl;

    tf_data.clear();
    tf_data.resize(256, glm::vec4(0.0f));

    // compute the maximum persistence from the pairs
    uint32_t maxPersistence = 0;
    for (const auto& p : pairs) {
        uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
        if (pers > maxPersistence)
            maxPersistence = pers;
    }
    if (maxPersistence == 0)
        maxPersistence = 1;

    for (const auto& p : pairs) 
    {
        uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
        float normalizedPers = float(pers) / float(maxPersistence);
        normalizedPers = std::min(normalizedPers, 1.0f);

        // map normalizedPers to a hue value
        // normalizedPers==1 map to hue 0 (red) and 0 to hue 240 (blue)
        float hue = (1.0f - normalizedPers) * 240.0f;
        float saturation = 1.0f;
        float value = 1.0f;
        glm::vec3 rgb = hsv2rgb(hue, saturation, value);

        // map the birth value to an index in [0,255]
        float normalizedBirth = (float(p.birth) - vol_min) / float(vol_max - vol_min);
        uint32_t index = static_cast<uint32_t>(normalizedBirth * 255.0f);
        index = std::clamp(index, 0u, 255u);

        tf_data[index] = glm::vec4(rgb, 1.0f);
        std::cout << "Pair (Birth=" << p.birth << ", Death=" << p.death << ", Persistence=" << pers << ") -> Normalized Pers=" << normalizedPers << ", Hue=" << hue << ", Mapped index=" << index << ", Color=(" << rgb.r << ", " << rgb.g << ", " << rgb.b << ", 1)" << std::endl;
    }
}

void TransferFunction::update_from_histogram(const Volume& volume, std::vector<glm::vec4>& tf_data)
{
    const int bins = 256;
    std::vector<int> histogram(bins, 0);
    
    // compute the histogram
    for (const auto& value : volume.data) 
    {
        histogram[value]++;
    }
    
    // find the maximum count for normalization
    int maxCount = *std::max_element(histogram.begin(), histogram.end());
    if (maxCount == 0)
        maxCount = 1;
    
    // build the transfer function
    // for each intensity create color (grayscale) and set the alpha based on the normalized histogram frequency
    tf_data.resize(bins);
    for (int i = 0; i < bins; ++i) 
    {
        float normalized = static_cast<float>(histogram[i]) / maxCount;
        float intensity = static_cast<float>(i) / 255.0f;
        tf_data[i] = glm::vec4(intensity, intensity, intensity, normalized);
    }
    
    std::cout << "Transfer function updated from histogram with " << tf_data.size() << " entries." << std::endl;
}


