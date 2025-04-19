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
  
  const Volume* get_volume() const { return volume; }

private:
  const VulkanMainContext& vmc;
  vk::DescriptorPool imgui_pool;
  MergeTree* merge_tree = nullptr;
  TransferFunction* transfer_function = nullptr;
  const Volume* volume = nullptr;
  ImTextureID persistence_texture_ID = (ImTextureID)0;
  std::function<void(const PersistencePair&)> on_pair_selected;

  float normalization_factor = 255.0f;
  const std::vector<PersistencePair>* persistence_pairs = nullptr;
  bool cache_dirty = true;
  bool show_dots = true;
  int max_points_to_show = 0;
  float diagram_zoom = 1.0f;
  float marker_size = 2.0f;
  std::vector<double> xs, ys;
  std::vector<float > pers;
  std::vector<ImVec2> dot_pos;
};
} // namespace ve