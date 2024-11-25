#pragma once

#include <vector>

#include "app_state.hpp"
#include "ray_marcher.hpp"
#include "ui.hpp"
#include "vk/storage.hpp"
#include "vk/swapchain.hpp"
#include "vk/vulkan_command_context.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/renderer.hpp"
#include "vk/synchronization.hpp"
#include "scene.hpp"

namespace ve
{
// manages the orchestration of rendering and computing
class WorkContext
{
public:
  WorkContext(const VulkanMainContext& vmc, VulkanCommandContext& vcc);
  void construct(AppState& app_state, const Scene& scene);
  void destruct();
  void draw_frame(AppState& app_state);
  vk::Extent2D recreate_swapchain(bool vsync);
  std::vector<float> get_result_values();

private:
  const VulkanMainContext& vmc;
  VulkanCommandContext& vcc;
  Storage storage;
  Swapchain swapchain;
  Renderer renderer;
  Ray_Marcher ray_marcher;
  uint32_t read_only_buffer_idx = 0;
  UI ui;
  std::vector<Synchronization> syncs;
  bool compute_finished = false;

  void render(uint32_t image_idx, AppState& app_state);
};
} // namespace ve
