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
  void set_raw_persistence_pairs(const std::vector<PersistencePair>& pairs);
  void set_gradient_persistence_pairs(const std::vector<PersistencePair>& pairs);
  void volume_highlight_persistence_pairs_gradient(const std::vector<std::pair<PersistencePair, float>>& pairs, int ramp_index);
  void highlight_diff(const PersistencePair &base, const PersistencePair &mask);
  void highlight_intersection(const PersistencePair &a, const PersistencePair &b);
  void highlight_union(const PersistencePair &a, const PersistencePair &b);
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
  uint32_t global_max_persistence = 1;
  const Volume* scalar_volume = nullptr;
  Volume gradient_volume;
  std::vector<glm::vec4> tf_data;
  std::vector<PersistencePair> persistence_pairs;
  std::vector<PersistencePair> raw_persistence_pairs;
  std::vector<Synchronization> syncs;
  std::vector<DeviceTimer> device_timers;
  std::vector<PersistencePair> gradient_persistence_pairs;
  std::vector<std::pair<PersistencePair, glm::vec4>> custom_colors;
  void render(uint32_t image_idx, AppState& app_state, uint32_t read_only_image);
  void apply_custom_color_to_volume(const std::vector<PersistencePair>& pairs, const ImVec4& color);
  void reset_custom_colors();
  void export_persistence_pairs_to_csv(const std::vector<PersistencePair>& scalar_pairs, const std::vector<PersistencePair>& gradient_pairs, const std::string& scalar_filename  = "scalar_pairs.csv", const std::string& gradient_filename = "gradient_pairs.csv") const;
};
} // namespace ve
