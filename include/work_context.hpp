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
#include <unordered_set>
#include "util/timer.hpp"

namespace ve {

class WorkContext
{
public:
  WorkContext(const VulkanMainContext& vmc, VulkanCommandContext& vcc, std::vector<PersistencePair> raw_pairs, std::vector<int> raw_filt, std::vector<PersistencePair> raw_grad_pairs, std::vector<int> raw_grad_filt);
  void construct(AppState& app_state, const Volume& volume);
  void destruct();
  void reload_shaders();
  void draw_frame(AppState& app_state);
  vk::Extent2D recreate_swapchain(bool vsync);
  void set_persistence_pairs(std::vector<PersistencePair> pairs, const Volume& volume);
  void load_persistence_diagram_texture(const std::string &filePath);
  void set_gradient_persistence_pairs(const std::vector<PersistencePair>& pairs);
  void volume_highlight_persistence_pairs(const std::vector<std::pair<PersistencePair, float>>& pairs, int ramp_index);
  void highlight_diff(const PersistencePair &base, const PersistencePair &mask);
  void highlight_intersection(const PersistencePair &a, const PersistencePair &b);
  void highlight_union(const PersistencePair &a, const PersistencePair &b);
  void highlight_onlyA(const PersistencePair& a, const PersistencePair& b, const ImVec4& col);
  void highlight_onlyB(const PersistencePair& a, const PersistencePair& b, const ImVec4& col);
  void reproject_and_compare();
  void fillTF2DFromVolume(const Volume& vol);
  std::vector<std::pair<int,int>> last_tf2d_bins;
  bool use_gradient_highlight = false;
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
  Timer<float> timer;
  using ms = std::milli;
  std::vector<std::pair<int,int>> click_bins;
  std::vector<ImU32> click_colors;
  int tf2dOverlayMode = 0; // 0 = Dots, 1 = Rects
  int pending_reproject_idx = -1;
  std::pair<int,int> last_scalar_range{0,0};
  std::pair<int,int> last_gradient_range{0,0};
  std::vector<std::vector<int>> grads_by_scalar;
  std::unordered_set<uint32_t> brush_seen;
  std::vector<glm::vec4> tf_data;
  std::vector<PersistencePair> persistence_pairs;
  std::vector<Synchronization> syncs;
  std::vector<DeviceTimer> device_timers;
  std::vector<PersistencePair> gradient_persistence_pairs;
  std::vector<PersistencePair> raw_persistence_pairs;
  std::vector<int> scalar_filtration;
  std::vector<PersistencePair> raw_gradient_pairs;
  std::vector<int> gradient_filtration;
  std::vector<std::pair<PersistencePair, glm::vec4>> custom_colors;
  void render(uint32_t image_idx, AppState& app_state, uint32_t read_only_image);
  void apply_custom_color_to_volume(const std::vector<PersistencePair>& pairs, const ImVec4& color);
  void reset_custom_colors();
  void export_persistence_pairs_to_csv(const std::vector<PersistencePair>& scalar_pairs, const std::vector<PersistencePair>& gradient_pairs, const std::string& scalar_filename  = "scalar_pairs.csv", const std::string& gradient_filename = "gradient_pairs.csv") const;
  std::pair<uint32_t, uint32_t> clamp_and_sort_range(const PersistencePair& p);
};
} // namespace ve
