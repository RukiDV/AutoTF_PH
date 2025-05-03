#pragma once

#include <vector>
#include "app_state.hpp"
#include "persistence.hpp"
#include "threshold_cut.hpp"
#include "ray_marcher.hpp"
#include "transfer_function.hpp"
#include "ui.hpp"
#include "vk/storage.hpp"
#include "vk/swapchain.hpp"
#include "vk/vulkan_command_context.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/renderer.hpp"
#include "vk/synchronization.hpp"
#include "volume.hpp"
#include "gpu_renderer.hpp"
#include "util/texture_loader.hpp"
#include "vk/device_timer.hpp"

namespace ve {

class WorkContext
{
public:
  WorkContext(const VulkanMainContext& vmc, VulkanCommandContext& vcc);
  void construct(AppState& app_state, const Volume& volume);
  void destruct();
  void reload_shaders();
  void draw_frame(AppState& app_state);
  vk::Extent2D recreate_swapchain(bool vsync);
  void set_persistence_pairs(const std::vector<PersistencePair>& pairs, const Volume& volume);
  void load_persistence_diagram_texture(const std::string &filePath);
  void highlight_persistence_pair(const PersistencePair& pair);
  void isolate_persistence_pairs(const std::vector<PersistencePair>& pairs);
  void volume_highlight_persistence_pairs(const std::vector<PersistencePair>& pairs);
  void set_raw_persistence_pairs(const std::vector<PersistencePair>& pairs);
  
private:  
  const VulkanMainContext& vmc;
  VulkanCommandContext& vcc;
  Storage storage;
  Swapchain swapchain;
  Renderer renderer;
  RayMarcher ray_marcher;
  TextureResourceImGui persistence_texture_resource;
  uint32_t read_only_buffer_idx = 0;
  UI ui;
  TransferFunction transfer_function;
  MergeTree merge_tree;
  std::vector<PersistencePair> persistence_pairs;
  std::vector<PersistencePair> raw_persistence_pairs;
  std::vector<Synchronization> syncs;
  std::vector<DeviceTimer> device_timers;

  void render(uint32_t image_idx, AppState& app_state, uint32_t read_only_image);
  void histogram_based_tf(const Volume &volume, std::vector<glm::vec4> &tf_data); 
  void refine_with_ph(const Volume &volume, int ph_threshold, std::vector<glm::vec4> &tf_data);
};
} // namespace ve
