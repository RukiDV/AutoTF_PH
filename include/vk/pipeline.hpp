#pragma once

#include "vk/common.hpp"
#include "vk/render_pass.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/shader.hpp"

namespace ve
{
class Pipeline
{
public:
  Pipeline(const VulkanMainContext& vmc);
  void construct(const RenderPass& render_pass, std::optional<vk::DescriptorSetLayout> set_layout, const std::vector<ShaderInfo>& shader_infos, const std::vector<vk::PushConstantRange>& pcrs = {});
  void construct(vk::DescriptorSetLayout set_layout, const ShaderInfo& shader_info, uint32_t push_constant_byte_size);
  void destruct();
  const vk::Pipeline& get() const;
  const vk::PipelineLayout& get_layout() const;

private:
  const VulkanMainContext& vmc;
  vk::PipelineLayout pipeline_layout;
  vk::Pipeline pipeline;
};
} // namespace ve
