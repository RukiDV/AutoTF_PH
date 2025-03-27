#pragma once

#include <vector>
#include "glm/vec4.hpp"
#include "persistence.hpp"
#include "volume.hpp"
#include <cmath>

// convert an hsv color h in [0, 360], s and v in [0,1] to an rgb color
inline glm::vec3 hsv2rgb(float h, float s, float v)
{
    float c = v * s;
    float h_prime = h / 60.0f;
    float x = c * (1.0f - fabsf(fmodf(h_prime, 2.0f) - 1.0f));
    glm::vec3 rgb(0.0f);
    
    if (0.0f <= h_prime && h_prime < 1.0f)
        rgb = glm::vec3(c, x, 0.0f);
    else if (1.0f <= h_prime && h_prime < 2.0f)
        rgb = glm::vec3(x, c, 0.0f);
    else if (2.0f <= h_prime && h_prime < 3.0f)
        rgb = glm::vec3(0.0f, c, x);
    else if (3.0f <= h_prime && h_prime < 4.0f)
        rgb = glm::vec3(0.0f, x, c);
    else if (4.0f <= h_prime && h_prime < 5.0f)
        rgb = glm::vec3(x, 0.0f, c);
    else if (5.0f <= h_prime && h_prime < 6.0f)
        rgb = glm::vec3(c, 0.0f, x);

    float m = v - c;
    return rgb + glm::vec3(m);
}

class TransferFunction 
{
public:
    void update(const std::vector<PersistencePair>& pairs, const Volume& volume, std::vector<glm::vec4>& tf_data);
    
    void update_from_histogram(const Volume& volume, std::vector<glm::vec4>& tf_data);
};