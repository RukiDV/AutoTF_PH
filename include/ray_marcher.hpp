#pragma once

#include "app_state.hpp"
#include "volume.hpp"
#include "vk/descriptor_set_handler.hpp"
#include "vk/pipeline.hpp"
#include "vk/storage.hpp"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace ve
{
class RayMarcher
{
public:
  RayMarcher(const VulkanMainContext& vmc, Storage& storage);
  void setup_storage(AppState& app_state, const Volume& volume, const Volume& gradient_volume);
  void construct(AppState& app_state, VulkanCommandContext& vcc, glm::uvec3 volume_resolution);
  void destruct();
  void reload_shaders();
  void compute(vk::CommandBuffer& cb, AppState& app_state, uint32_t read_only_buffer_idx);

private:
  enum Buffers
  {
    RAY_MARCHER_BUFFER_0 = 0,
    RAY_MARCHER_BUFFER_1 = 1,
    VOLUME_BUFFER = 2,
    TF_BUFFER = 3,
    UNIFORM_BUFFER = 4,
    GRADIENT_VOLUME_BUFFER = 5,
    BUFFER_COUNT
  };

  enum Images
  {
    RAY_MARCHER_IMAGE = 0,
    IMAGE_COUNT
  };

  const VulkanMainContext& vmc;
  Storage& storage;
  Pipeline clear_pipeline;
  Pipeline pipeline;
  DescriptorSetHandler dsh;
  std::array<int32_t, BUFFER_COUNT> buffers;
  std::array<int32_t, IMAGE_COUNT> images;

  struct PushConstants
  {
    uint32_t display_mode = 0;
    float max_gradient = 0.0f;
  } pc;

  void create_pipeline(const AppState& app_state, glm::uvec3 volume_resolution);
  void create_descriptor_set();
  void clear_storage_indices();
};
} // namespace ve
