#pragma once

#include "app_state.hpp"
#include "vk/descriptor_set_handler.hpp"
#include "vk/pipeline.hpp"
#include "vk/storage.hpp"
#include "glm/vec2.hpp"

namespace ve
{
class Renderer
{
public:
  Renderer(const VulkanMainContext& vmc, Storage& storage);
  void setup_storage(AppState& app_state);
  void construct(const RenderPass& render_pass, const AppState& app_state);
  void destruct();
  void render(vk::CommandBuffer& cb, AppState& app_state, uint32_t read_only_buffer_idx, const vk::Framebuffer& framebuffer, const vk::RenderPass& render_pass);

private:
  enum Images
  {
    RENDER_IMAGE = 0,
    IMAGE_COUNT
  };

  const VulkanMainContext& vmc;
  Storage& storage;
  Pipeline pipeline;
  DescriptorSetHandler dsh;
  std::array<int32_t, IMAGE_COUNT> images;

  void create_pipeline(const RenderPass& render_pass, const AppState& app_state);
  void create_descriptor_set();
};
} // namespace ve
