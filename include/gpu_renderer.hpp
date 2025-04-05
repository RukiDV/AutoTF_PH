#pragma once

#include "volume.hpp"
#include "persistence.hpp"

std::vector<PersistencePair> calculate_persistence_pairs(const Volume& volume, std::vector<int>& filtration_values, FiltrationMode mode = FiltrationMode::LowerStar);

int gpu_render(const Volume& volume);

Volume create_test_volume_gradient();