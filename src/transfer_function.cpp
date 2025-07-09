#include "transfer_function.hpp"
#include "app_state.hpp"
#include "glm/vec3.hpp"
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

std::pair<uint32_t, uint32_t> TransferFunction::compute_min_max_scalar(const Volume& volume)
{
  // parallel reduction to compute min and max scalar value in the volume
  uint32_t min_value = std::numeric_limits<uint32_t>::max();
  uint32_t max_value = std::numeric_limits<uint32_t>::min();

  #pragma omp parallel for reduction(min: min_value) reduction(max: max_value)
  for (size_t i = 0; i < volume.data.size(); ++i)
  {
    uint32_t v = volume.data[i];
    min_value = std::min(min_value, v);
    max_value = std::max(max_value, v);
  }
  return {min_value, max_value};
}

void TransferFunction::update(const std::vector<PersistencePair>& pairs, const Volume& volume, std::vector<glm::vec4>& tf_data)
{
  // compute volume scalar range
  auto [vol_min, vol_max] = compute_min_max_scalar(volume);
  float span = float(vol_max > vol_min ? (vol_max - vol_min) : 1);

  tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));
  for (int s = 0; s < AppState::TF2D_BINS; ++s)
  {
    float normalized = float(s) / float(AppState::TF2D_BINS - 1);
    glm::vec4 base_col(normalized, normalized, normalized, 1.0f);
    for (int g = 0; g < AppState::TF2D_BINS; ++g)
    {
        tf_data[g * AppState::TF2D_BINS + s] = base_col;
    }
  }

  // find max persistence in parallel
  uint32_t max_pers = 1;
  #pragma omp parallel for reduction(max: max_pers)
  for (size_t i = 0; i < pairs.size(); ++i)
  {
    const auto &p = pairs[i];
    uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
    max_pers = std::max(max_pers, pers);
  }

  // precompute brush entries (bi, di, rgb)
  struct Brush
  {
    uint32_t bi, di; 
    glm::vec3 rgb;
  };

  std::vector<Brush> brushes;
  brushes.reserve(pairs.size());
  for (const auto &p : pairs)
  {
    uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
    float norm_p = float(pers) / float(max_pers);
    float hue    = (1.0f - norm_p) * 240.0f;
    glm::vec3 rgb  = hsv2rgb(hue, 1.0f, 1.0f);

    float nb = (float(p.birth) - float(vol_min)) / span;
    float nd = (float(p.death) - float(vol_min)) / span;
    uint32_t bi = uint32_t(std::clamp(nb, 0.0f, 1.0f) * float(AppState::TF2D_BINS - 1));
    uint32_t di = uint32_t(std::clamp(nd, 0.0f, 1.0f) * float(AppState::TF2D_BINS - 1));
    if (bi > di) std::swap(bi, di);

    brushes.push_back({bi, di, rgb});
  }

  // parallel paint by row
  #pragma omp parallel for schedule(static)
  for (int g = 0; g < AppState::TF2D_BINS; ++g)
  {
    auto row_begin = tf_data.begin() + g * AppState::TF2D_BINS;
    for (const auto &br : brushes)
    {
      auto it = row_begin + br.bi;
      auto it_end = row_begin + br.di + 1;
      std::fill(it, it_end, glm::vec4(br.rgb, 1.0f));
    }
  }
}