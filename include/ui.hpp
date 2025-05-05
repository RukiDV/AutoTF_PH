#pragma once

#include "app_state.hpp"
#include "vk/render_pass.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/vulkan_command_context.hpp"
#include "merge_tree.hpp"
#include "transfer_function.hpp"
#include <functional>
#include "imgui.h"
#include <vector>

namespace ve
{
class UI
{
public:
  explicit UI(const VulkanMainContext& vmc);
  void construct(VulkanCommandContext& vcc, const RenderPass& render_pass, uint32_t frames);
  void destruct();
  void draw(vk::CommandBuffer& cb, AppState& app_state);

  void set_transfer_function(TransferFunction* transfer_function);
  void set_volume(const Volume* volume);
  void set_persistence_pairs(const std::vector<PersistencePair>* pairs);
  void set_persistence_texture(ImTextureID tex);
  void set_on_pair_selected(const std::function<void(const PersistencePair&)>& callback);
  void set_on_range_applied(std::function<void(const std::vector<PersistencePair>&)> cb);
  void set_on_multi_selected(const std::function<void(const std::vector<PersistencePair>&)>& cb);
  void set_on_brush_selected(const std::function<void(const std::vector<PersistencePair>&)>& cb);
  
  const Volume* get_volume() const { return volume; }

private:
  const VulkanMainContext& vmc;
  vk::DescriptorPool imgui_pool;
  MergeTree* merge_tree = nullptr;
  TransferFunction* transfer_function = nullptr;
  const Volume* volume = nullptr;
  ImTextureID persistence_texture_ID = (ImTextureID)0;
  std::function<void(const PersistencePair&)> on_pair_selected;
  std::function<void(const std::vector<PersistencePair>&)> on_range_applied;

  bool cache_dirty = true;
  bool show_dots = true;
  bool range_active = false;
  int max_points_to_show = 0;
  float normalization_factor = 255.0f;
  float diagram_zoom = 1.0f;
  float marker_size = 2.0f;
  float birth_range[2] = { 0.0f, 255.0f};
  float death_range[2] = { 0.0f, 255.0f};
  float persistence_range[2] = { 0.0f, 255.0f};
  float blink_timer = 0.0f;
  const std::vector<PersistencePair>* persistence_pairs = nullptr;
  std::vector<double> xs, ys;
  std::vector<float > pers;
  std::vector<ImVec2> dot_pos;
  int selected_idx = -1; // –1 means “no selection”
  ImU32  selected_color = IM_COL32(255,0,255,255);

  std::vector<int> multi_selected_idxs;
  std::vector<ImU32> multi_selected_cols;
  bool brush_active = false;
  ImVec2 brush_start;
  ImVec2 brush_end;

  std::function<void(const std::vector<PersistencePair>&)> on_multi_selected;
  std::function<void(const std::vector<PersistencePair>&)> on_brush_selected;
};
} // namespace ve