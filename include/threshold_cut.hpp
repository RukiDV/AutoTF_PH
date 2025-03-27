#pragma once

#include "transfer_function.hpp"
#include <vector>
#include <glm/vec4.hpp>
#include <vector>
#include "persistence.hpp"

std::vector<PersistencePair> threshold_cut(const std::vector<PersistencePair>& pairs, uint32_t threshold); 