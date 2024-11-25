#pragma once

#include "app_state.hpp"
#include "vk/render_pass.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/vulkan_command_context.hpp"

namespace ve
{
class UI
{
public:
  explicit UI(const VulkanMainContext& vmc);
  void construct(VulkanCommandContext& vcc, const RenderPass& render_pass, uint32_t frames);
  void destruct();
  void draw(vk::CommandBuffer& cb, AppState& app_state);
private:
  const VulkanMainContext& vmc;
  vk::DescriptorPool imgui_pool;
};
} // namespace ve
