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

  void set_merge_tree(MergeTree* merge_tree);
  void set_transfer_function(TransferFunction* transfer_function);
  void set_volume(const Volume* volume);
  void set_persistence_pairs(const std::vector<PersistencePair>* pairs);
  void set_persistence_texture(ImTextureID tex)
  {
    persistence_texture_ID = tex;
  }
  void set_on_pair_selected(const std::function<void(const PersistencePair&)>& callback)
  {
      on_pair_selected = callback;
  }
  const Volume* get_volume() const { return volume; }

  void set_persistent_dots(const std::vector<ImVec2>& dots);
  void set_normalization_factor(float nf);
private:
  const VulkanMainContext& vmc;
  vk::DescriptorPool imgui_pool;
  MergeTree* merge_tree = nullptr;
  TransferFunction* transfer_function = nullptr;
  const Volume* volume = nullptr;
  const std::vector<PersistencePair>* persistence_pairs = nullptr;
  ImTextureID persistence_texture_ID = (ImTextureID)0;
  std::function<void(const PersistencePair&)> on_pair_selected;

  std::vector<ImVec2> persistentDots;
  float normalizationFactor = 255.0f;
  float diagramZoom = 1.0f;
};
} // namespace ve