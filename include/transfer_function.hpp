#pragma once

#include <vector>
#include "glm/vec4.hpp"
#include "persistence.hpp"
#include "volume.hpp"
#include "vk/storage.hpp"

class TransferFunction 
{
public:
    TransferFunction() = default;

    void update(const std::vector<PersistencePair>& pairs, const Volume& volume, std::vector<glm::vec4>& tf_data);
private:
};