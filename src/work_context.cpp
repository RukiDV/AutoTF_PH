#include "work_context.hpp"
#include <vulkan/vulkan_enums.hpp>

namespace ve
{
WorkContext::WorkContext(const VulkanMainContext& vmc, VulkanCommandContext& vcc)
  : vmc(vmc), vcc(vcc), storage(vmc, vcc), swapchain(vmc, vcc, storage), renderer(vmc, storage), ray_marcher(vmc, storage), ui(vmc)
{}

void WorkContext::construct(AppState& app_state, const Scene& scene)
{
  vcc.add_graphics_buffers(frames_in_flight);
  vcc.add_compute_buffers(2);
  vcc.add_transfer_buffers(1);
  renderer.setup_storage(app_state);
  ray_marcher.setup_storage(app_state, scene);
  swapchain.construct(app_state.vsync);
  app_state.get_window_extent() = swapchain.get_extent();
  for (uint32_t i = 0; i < frames_in_flight; ++i)
  {
    syncs.emplace_back(vmc.logical_device.get());
  }
  const glm::uvec2 resolution(app_state.get_render_extent().width, app_state.get_render_extent().height);
  renderer.construct(swapchain.get_render_pass(), app_state);
  ray_marcher.construct(app_state, vcc, scene.resolution);
  ui.construct(vcc, swapchain.get_render_pass(), frames_in_flight);
}

void WorkContext::destruct()
{
  vmc.logical_device.get().waitIdle();
  for (auto& sync : syncs) sync.destruct();
  syncs.clear();
  swapchain.destruct();
  renderer.destruct();
  ray_marcher.destruct();
  ui.destruct();
  storage.clear();
}

void WorkContext::draw_frame(AppState &app_state)
{
  syncs[app_state.current_frame].wait_for_fence(Synchronization::F_RENDER_FINISHED);
  syncs[app_state.current_frame].reset_fence(Synchronization::F_RENDER_FINISHED);
  vk::ResultValue<uint32_t> image_idx = vmc.logical_device.get().acquireNextImageKHR(swapchain.get(), uint64_t(-1), syncs[app_state.current_frame].get_semaphore(Synchronization::S_IMAGE_AVAILABLE));
  VE_CHECK(image_idx.result, "Failed to acquire next image!");
  render(image_idx.value, app_state);
  app_state.current_frame = (app_state.current_frame + 1) % frames_in_flight;
  app_state.total_frames++;
}

vk::Extent2D WorkContext::recreate_swapchain(bool vsync)
{
  vmc.logical_device.get().waitIdle();
  swapchain.recreate(vsync);
  return swapchain.get_extent();
}

void WorkContext::render(uint32_t image_idx, AppState& app_state)
{
  // dispatch next compute iteration as soon as the previous one is done
  if (syncs[0].is_fence_finished(Synchronization::F_COMPUTE_FINISHED))
  {
    // update index of buffer that is now safe to read
    syncs[0].reset_fence(Synchronization::F_COMPUTE_FINISHED);

    // wait for previous rendering job that uses the buffer that will now be written
    syncs[1 - app_state.current_frame].wait_for_fence(Synchronization::F_RENDER_FINISHED);

    vk::CommandBuffer& cb = vcc.get_one_time_transfer_buffer();

    Image& render_texture = storage.get_image_by_name("render_texture");
    Image& ray_marcher_output_texture = storage.get_image_by_name("ray_marcher_output_texture");

    perform_image_layout_transition(cb, render_texture.get_image(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferWrite, 0, 1, 1);
    perform_image_layout_transition(cb, ray_marcher_output_texture.get_image(), vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eTransferRead, 0, 1, 1);

    copy_image(cb, ray_marcher_output_texture.get_image(), render_texture.get_image(), app_state.get_render_extent().width, app_state.get_render_extent().height, 1);

    perform_image_layout_transition(cb, render_texture.get_image(), vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead, 0, 1, 1);
    perform_image_layout_transition(cb, ray_marcher_output_texture.get_image(), vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eMemoryWrite, 0, 1, 1);
    cb.end();

    syncs[0].reset_fence(Synchronization::F_COPY_FINISHED);
    vk::SubmitInfo si(0, nullptr, nullptr, 1, &cb);
    vmc.get_transfer_queue().submit(si, syncs[0].get_fence(Synchronization::F_COPY_FINISHED));
    syncs[0].wait_for_fence(Synchronization::F_COPY_FINISHED);
    syncs[0].reset_fence(Synchronization::F_COPY_FINISHED);
    
    vk::CommandBuffer& compute_cb = vcc.begin(vcc.compute_cbs[1]);
    ray_marcher.compute(compute_cb, app_state, read_only_buffer_idx);
    compute_cb.end();
    read_only_buffer_idx = (read_only_buffer_idx + 1) % frames_in_flight;

    vk::SubmitInfo compute_si(0, nullptr, nullptr, 1, &vcc.compute_cbs[1]);
    vmc.get_compute_queue().submit(compute_si, syncs[0].get_fence(Synchronization::F_COMPUTE_FINISHED));
  }

  vk::CommandBuffer& cb = vcc.begin(vcc.graphics_cbs[app_state.current_frame]);
  Image& render_texture = storage.get_image_by_name("render_texture");
  if (render_texture.get_layout() != vk::ImageLayout::eShaderReadOnlyOptimal)
  {
    perform_image_layout_transition(cb, render_texture.get_image(), render_texture.get_layout(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eMemoryRead, 0, 1, 1);
    render_texture.set_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
  }
  renderer.render(cb, app_state, read_only_buffer_idx, swapchain.get_framebuffer(image_idx), swapchain.get_render_pass().get());
  if (app_state.show_ui) ui.draw(cb, app_state);
  cb.endRenderPass();
  cb.end();

  std::vector<vk::Semaphore> render_wait_semaphores;
  std::vector<vk::PipelineStageFlags> render_wait_stages;
  render_wait_semaphores.push_back(syncs[app_state.current_frame].get_semaphore(Synchronization::S_IMAGE_AVAILABLE));
  render_wait_stages.push_back(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  std::vector<vk::Semaphore> render_signal_semaphores;
  render_signal_semaphores.push_back(syncs[app_state.current_frame].get_semaphore(Synchronization::S_RENDER_FINISHED));
  vk::SubmitInfo render_si(render_wait_semaphores.size(), render_wait_semaphores.data(), render_wait_stages.data(), 1, &vcc.graphics_cbs[app_state.current_frame], render_signal_semaphores.size(), render_signal_semaphores.data());
  vmc.get_graphics_queue().submit(render_si, syncs[app_state.current_frame].get_fence(Synchronization::F_RENDER_FINISHED));

  vk::PresentInfoKHR present_info(1, &syncs[app_state.current_frame].get_semaphore(Synchronization::S_RENDER_FINISHED), 1, &swapchain.get(), &image_idx);
  VE_CHECK(vmc.get_present_queue().presentKHR(present_info), "Failed to present image!");
}
} // namespace ve
