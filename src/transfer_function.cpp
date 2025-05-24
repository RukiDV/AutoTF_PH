#include "transfer_function.hpp"
#include "glm/vec3.hpp"
#include <algorithm>
#include <limits>

std::pair<uint32_t, uint32_t> TransferFunction::compute_min_max_scalar(const Volume& volume) 
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
    // compute the scalar range of this volume
    auto [vol_min, vol_max] = compute_min_max_scalar(volume);
    float span = float(vol_max > vol_min ? (vol_max - vol_min) : 1);

    tf_data.assign(256, glm::vec4(0.0f));

    // find the maximum persistence
    uint32_t max_pers = 1;
    for (auto &p : pairs)
    {
        uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
        max_pers = std::max(max_pers, pers);
    }

    for (auto &p : pairs)
    {
        uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
        float norm_p = float(pers) / float(max_pers);
        float hue = (1.0f - norm_p) * 240.0f;
        glm::vec3 rgb = hsv2rgb(hue, 1.0f, 1.0f);

        float nb = (float(p.birth) - float(vol_min)) / span;
        float nd = (float(p.death) - float(vol_min)) / span;
        uint32_t bi = uint32_t(std::clamp(nb, 0.0f, 1.0f) * 255.0f);
        uint32_t di = uint32_t(std::clamp(nd, 0.0f, 1.0f) * 255.0f);
        if (bi > di) std::swap(bi, di);

        for (uint32_t i = bi; i <= di; ++i)
        {
            tf_data[i] = glm::vec4(rgb, 1.0f);
        }
    }
}