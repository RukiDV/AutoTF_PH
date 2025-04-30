#include "vk/renderer.hpp"

namespace ve
{
Renderer::Renderer(const VulkanMainContext& vmc, Storage& storage) : vmc(vmc), storage(storage), pipeline(vmc), dsh(vmc, frames_in_flight)
{}

void Renderer::setup_storage(AppState& app_state)
{
  std::vector<unsigned char> initial_image(app_state.get_render_extent().width * app_state.get_render_extent().height * 4, 0);
  images[RENDER_IMAGE] = storage.add_image("render_texture", initial_image.data(), app_state.get_render_extent().width, app_state.get_render_extent().height, false, 0, QueueFamilyFlags::Graphics | QueueFamilyFlags::Transfer | QueueFamilyFlags::Compute, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage);
}

void Renderer::construct(const RenderPass& render_pass, const AppState& app_state)
{
  std::cout << "Constructing renderer" << std::endl;
  create_descriptor_set();
  create_pipeline(render_pass, app_state);
  std::cout << "Successfully constructed renderer" << std::endl;
}

void Renderer::destruct()
{
  for (int32_t i : images)
  {
    if (i > -1) storage.destroy_image(i);
  }
  pipeline.destruct();
  dsh.destruct();
}

void Renderer::create_pipeline(const RenderPass& render_pass, const AppState& app_state)
{
  std::vector<ShaderInfo> render_shader_infos(2);
  render_shader_infos[0] = ShaderInfo{"image.vert", vk::ShaderStageFlagBits::eVertex};
  render_shader_infos[1] = ShaderInfo{"image.frag", vk::ShaderStageFlagBits::eFragment};
  pipeline.construct(render_pass, dsh.get_layout(), render_shader_infos);
}

void Renderer::create_descriptor_set()
{
  dsh.add_binding(0, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment);
  for (uint32_t i = 0; i < frames_in_flight; ++i)
  {
    dsh.add_descriptor(i, 0, storage.get_image_by_name("render_texture"));
  }
  dsh.construct();
}

void Renderer::render(vk::CommandBuffer& cb, AppState& app_state, uint32_t read_only_buffer_idx, const vk::Framebuffer& framebuffer, const vk::RenderPass& render_pass)
{
  vk::RenderPassBeginInfo rpbi{};
  rpbi.sType = vk::StructureType::eRenderPassBeginInfo;
  rpbi.renderPass = render_pass;
  rpbi.framebuffer = framebuffer;
  rpbi.renderArea.offset = vk::Offset2D(0, 0);
  rpbi.renderArea.extent = app_state.get_window_extent();
  std::vector<vk::ClearValue> clear_values(2);
  clear_values[0].color = vk::ClearColorValue(1.0f, 1.0f, 1.0f, 1.0f);
  clear_values[1].depthStencil.depth = 1.0f;
  clear_values[1].depthStencil.stencil = 0;
  rpbi.clearValueCount = clear_values.size();
  rpbi.pClearValues = clear_values.data();
  cb.beginRenderPass(rpbi, vk::SubpassContents::eInline);

  vk::Viewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = app_state.get_window_extent().width;
  viewport.height = app_state.get_window_extent().height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  cb.setViewport(0, viewport);
  vk::Rect2D scissor{};
  scissor.offset = vk::Offset2D(0, 0);
  scissor.extent = app_state.get_window_extent();
  cb.setScissor(0, scissor);

  cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.get_layout(), 0, {dsh.get_sets()[read_only_buffer_idx]}, {});
  cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.get());
  cb.draw(3, 1, 0, 0);
}
} // namespace ve
