#pragma once

#include "volume.hpp"
#include "persistence.hpp"

int gpu_render(const Volume& volume, const std::vector<PersistencePair>& pairs);
