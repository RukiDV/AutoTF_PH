#include "transfer_function.hpp"
#include <algorithm>

#include "glm/vec4.hpp"
#include "util/vec_streams.hpp"

bool compare_persistence(const PersistencePair& a, const PersistencePair& b) 
{
    return a.persistence() > b.persistence();
}

PersistencePair find_most_persistent(const std::vector<PersistencePair>& pairs) 
{
    return *std::max_element(pairs.begin(), pairs.end(), compare_persistence);
}

std::vector<PersistencePair> find_top_n_persistent(const std::vector<PersistencePair>& pairs, size_t n) 
{
    std::vector<PersistencePair> sorted_pairs = pairs;

    // sort by persistence descending order
    std::sort(sorted_pairs.begin(), sorted_pairs.end(), compare_persistence);

    // resize to keep only the top n pairs
    if (sorted_pairs.size() > n) 
    {
        sorted_pairs.erase(sorted_pairs.begin() + n, sorted_pairs.end());
    }
    return sorted_pairs;
}

std::pair<uint32_t, uint32_t> compute_min_max_scalar(const Volume& volume) 
{
    uint32_t min_value = std::numeric_limits<uint32_t>::max();
    uint32_t max_value = std::numeric_limits<uint32_t>::min();

    for (uint32_t value : volume.data) 
    {
        if (value < min_value) min_value = value;
        if (value > max_value) max_value = value;
    }
    return {min_value, max_value};
}

void TransferFunction::update(const std::vector<PersistencePair>& pairs, const Volume& volume, std::vector<glm::vec4>& tf_data) 
{
    auto [min_value_volume, max_value_volume] = compute_min_max_scalar(volume);
    std::cout << "Min value volume: " << static_cast<int>(min_value_volume) << ", Max value volume: " << static_cast<int>(max_value_volume) << std::endl;

    tf_data.clear();
    tf_data.resize(256, glm::vec4(0.0f)); // unvisible

    std::vector<PersistencePair> top_pairs = find_top_n_persistent(pairs, 5);
    std::cout << "Top 5 persistent pairs:" << std::endl;
    for (const auto& pair : top_pairs) 
    {
        std::cout << "Birth: " << pair.birth << ", Death: " << pair.death << ", Persistence: " << pair.persistence() << std::endl;
    }

    uint32_t min_value = std::numeric_limits<uint32_t>::max();
    uint32_t max_value = std::numeric_limits<uint32_t>::min();

    for (const auto& pair : top_pairs) 
    {
        if (pair.birth < min_value) min_value = pair.birth;
        if (pair.death > max_value) max_value = pair.death;
    }

    std::cout << "Top Pairs Min value: " << min_value << ", Top Pairs Max value: " << max_value << std::endl;

    std::vector<glm::vec4> colors = 
    {
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)
    };

    for (size_t idx = 0; idx < top_pairs.size(); ++idx) 
    {
        const auto& pair = top_pairs[idx];

        float normalized_birth = (static_cast<float>(pair.birth) - min_value) / (max_value - min_value);
        float normalized_death = (static_cast<float>(pair.death) - min_value) / (max_value - min_value);

        uint32_t birth = static_cast<uint32_t>(normalized_birth * 255);
        uint32_t death = static_cast<uint32_t>(normalized_death * 255);

        // limit indicies on valid area
        if (birth > 255) birth = 255;
        if (death > 255) death = 255;

        std::cout << "Raw birth: " << pair.birth << ", Raw death: " << pair.death << std::endl;
        std::cout << "Normalized birth: " << normalized_birth << ", Normalized death: " << normalized_death << std::endl;

        // select color from list
        glm::vec4 color = colors[idx % colors.size()];

        for (uint32_t i = birth; i <= death && i < tf_data.size(); ++i) 
        {
            tf_data[i] = color;
            std::cout << "Setting TF value " << i << " to color " << color.r << ", " << color.g << ", " << color.b << ", " << color.a << std::endl;
        }
    }

    for (size_t i = 0; i < tf_data.size(); ++i) 
    {
        if (tf_data[i].w > 0.0f) // visible values 
        {
            std::cout << "Transfer Function Value " << i << ": " << tf_data[i].r << ", " << tf_data[i].g << ", " << tf_data[i].b << ", " << tf_data[i].a << std::endl;
        }
    }
}