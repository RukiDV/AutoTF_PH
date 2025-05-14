#include "work_context.hpp"
#include <vulkan/vulkan_enums.hpp>
#include "stb/stb_image.h"
#include "transfer_function.hpp"

namespace ve
{
WorkContext::WorkContext(const VulkanMainContext& vmc, VulkanCommandContext& vcc)
  : vmc(vmc), vcc(vcc), storage(vmc, vcc), swapchain(vmc, vcc, storage), renderer(vmc, storage), ray_marcher(vmc, storage), persistence_texture_resource(vmc, storage), ui(vmc)
{}

void WorkContext::construct(AppState& app_state, const Volume& volume)
{
  vcc.add_graphics_buffers(frames_in_flight);
  vcc.add_compute_buffers(2);
  vcc.add_transfer_buffers(1);
  renderer.setup_storage(app_state);
  ray_marcher.setup_storage(app_state, volume);
  swapchain.construct(app_state.vsync);
  app_state.set_window_extent(swapchain.get_extent());
  for (uint32_t i = 0; i < frames_in_flight; ++i)
  {
    syncs.emplace_back(vmc.logical_device.get());
		device_timers.emplace_back(vmc);
  }
  const glm::uvec2 resolution(app_state.get_render_extent().width, app_state.get_render_extent().height);
  renderer.construct(swapchain.get_render_pass(), app_state);
  ray_marcher.construct(app_state, vcc, volume.resolution);
  ui.construct(vcc, swapchain.get_render_pass(), frames_in_flight);
  ui.set_transfer_function(&transfer_function);
  ui.set_volume(&volume);
  ui.set_persistence_pairs(&persistence_pairs);

  Volume grad_vol = compute_gradient_volume(volume);

  // calculate its persistence pairs
  std::vector<int> grad_filtration_values;
  auto raw_grad_pairs = calculate_persistence_pairs(grad_vol, grad_filtration_values, app_state.filtration_mode);

  gradient_persistence_pairs.clear();
  gradient_persistence_pairs.reserve(raw_grad_pairs.size());
  for (auto &p : raw_grad_pairs)
  {
      // look up the true gradient‐magnitude from the filtration array
      uint32_t b = grad_filtration_values[p.birth];
      uint32_t d = grad_filtration_values[p.death];
      gradient_persistence_pairs.emplace_back(b, d);
  }

  ui.set_gradient_persistence_pairs(&gradient_persistence_pairs);

  load_persistence_diagram_texture("output_plots/persistence_diagram.png");
  
  ui.set_on_pair_selected([this](const PersistencePair& hit){
    // iso-surface mode
    this->isolate_persistence_pairs({ hit });
  });

  ui.set_on_range_applied([this](const std::vector<PersistencePair>& range){
      if (!range.empty()) 
          // volume-highlight mode
          this->volume_highlight_persistence_pairs(range);
  });

  ui.set_on_multi_selected([this,&app_state](const std::vector<PersistencePair>& hits){
    if (app_state.display_mode == 0)
      this->isolate_persistence_pairs(hits);
    else
      this->volume_highlight_persistence_pairs(hits);
  });

  ui.set_on_brush_selected([this](const std::vector<PersistencePair>& hits){
      this->volume_highlight_persistence_pairs(hits);
  });
}

void WorkContext::destruct()
{
  vmc.logical_device.get().waitIdle();
  for (auto& sync : syncs) sync.destruct();
  for (auto& device_timer : device_timers) device_timer.destruct();
  syncs.clear();
  persistence_texture_resource.destruct();
  swapchain.destruct();
  renderer.destruct();
  ray_marcher.destruct();
  ui.destruct();
  storage.clear();
}

void WorkContext::reload_shaders()
{
  vmc.logical_device.get().waitIdle();
  ray_marcher.reload_shaders();
}

void WorkContext::draw_frame(AppState &app_state)
{
  syncs[app_state.current_frame].wait_for_fence(Synchronization::F_RENDER_FINISHED);
  syncs[app_state.current_frame].reset_fence(Synchronization::F_RENDER_FINISHED);
  if (app_state.total_frames > frames_in_flight)
	{
    if (app_state.current_frame == 0)
    {
      // update device timers
      for (int i = 0; i < DeviceTimer::TIMER_COUNT; i++) app_state.device_timings[i] = device_timers[app_state.current_frame].get_result_by_idx(i);
    }
	}

  vk::ResultValue<uint32_t> image_idx = vmc.logical_device.get().acquireNextImageKHR(swapchain.get(), uint64_t(-1), syncs[app_state.current_frame].get_semaphore(Synchronization::S_IMAGE_AVAILABLE));
  VE_CHECK(image_idx.result, "Failed to acquire next image!");

  uint32_t read_only_image = (app_state.total_frames / frames_in_flight) % frames_in_flight;
  if (app_state.save_screenshot)
  {
    storage.get_image_by_name("ray_marcher_output_texture").save_to_file(vcc);
    app_state.save_screenshot = false;
  }
  render(image_idx.value, app_state, read_only_image);
  app_state.current_frame = (app_state.current_frame + 1) % frames_in_flight;
  app_state.total_frames++;
}

vk::Extent2D WorkContext::recreate_swapchain(bool vsync)
{
  vmc.logical_device.get().waitIdle();
  swapchain.recreate(vsync);
  return swapchain.get_extent();
}

void WorkContext::render(uint32_t image_idx, AppState& app_state, uint32_t read_only_image)
{
  // dispatch next compute iteration as soon as the previous one is done
  if (app_state.current_frame == 0)
  {
    syncs[0].wait_for_fence(Synchronization::F_COMPUTE_FINISHED);
    // reset fence for the next compute iteration
    syncs[0].reset_fence(Synchronization::F_COMPUTE_FINISHED);

    app_state.cam.update_data();
    storage.get_buffer_by_name("ray_marcher_uniform_buffer").update_data_bytes(&app_state.cam.data, sizeof(Camera::Data));

    // wait for the previous rendering job that uses the buffer that will now be written
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

    vk::CommandBuffer& compute_cb = vcc.begin(vcc.compute_cbs[0]);
    device_timers[app_state.current_frame].reset(compute_cb, {DeviceTimer::VOLUME});
    device_timers[app_state.current_frame].start(compute_cb, DeviceTimer::VOLUME, vk::PipelineStageFlagBits::eComputeShader);
    ray_marcher.compute(compute_cb, app_state, read_only_buffer_idx);
    device_timers[app_state.current_frame].stop(compute_cb, DeviceTimer::VOLUME, vk::PipelineStageFlagBits::eComputeShader);
    compute_cb.end();
    read_only_buffer_idx = (read_only_buffer_idx + 1) % frames_in_flight;

    vk::SubmitInfo compute_si(0, nullptr, nullptr, 1, &vcc.compute_cbs[0]);
    vmc.get_compute_queue().submit(compute_si, syncs[0].get_fence(Synchronization::F_COMPUTE_FINISHED));
  }

  vk::CommandBuffer& graphics_cb = vcc.begin(vcc.graphics_cbs[app_state.current_frame]);
  device_timers[app_state.current_frame].reset(graphics_cb, {DeviceTimer::UI});
  Image& render_texture = storage.get_image_by_name("render_texture");
  if (render_texture.get_layout() != vk::ImageLayout::eShaderReadOnlyOptimal)
  {
    perform_image_layout_transition(graphics_cb, render_texture.get_image(), render_texture.get_layout(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eMemoryRead, 0, 1, 1);
    render_texture.set_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
  }
  renderer.render(graphics_cb, app_state, read_only_buffer_idx, swapchain.get_framebuffer(image_idx), swapchain.get_render_pass().get());
  device_timers[app_state.current_frame].start(graphics_cb, DeviceTimer::UI, vk::PipelineStageFlagBits::eTopOfPipe);
  if (app_state.show_ui) ui.draw(graphics_cb, app_state);
  device_timers[app_state.current_frame].stop(graphics_cb, DeviceTimer::UI, vk::PipelineStageFlagBits::eBottomOfPipe);
  graphics_cb.endRenderPass();
  graphics_cb.end();

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

void WorkContext::set_persistence_pairs(const std::vector<PersistencePair>& pairs, const Volume& volume) {
    persistence_pairs = pairs; 

    std::vector<glm::vec4> tf_data;
    transfer_function.update(persistence_pairs, volume, tf_data);
    std::cout << "Transfer function updated with " << tf_data.size() << " entries." << std::endl;
    storage.get_buffer_by_name("transfer_function").update_data_bytes(tf_data.data(), sizeof(glm::vec4) * tf_data.size());
    vmc.logical_device.get().waitIdle();
}

// build a histogram-based color/opacity array
  void WorkContext::histogram_based_tf(const Volume &volume, std::vector<glm::vec4> &tf_data)
  {
      const int bins = 256;
      std::vector<int> histogram(bins, 0);

      for (auto val : volume.data)
      {
          histogram[val]++;
      }

      int maxCount = *std::max_element(histogram.begin(), histogram.end());
      if (maxCount == 0) maxCount = 1;

      tf_data.resize(bins);
      for (int i = 0; i < bins; ++i)
      {
          float normalized = float(histogram[i]) / float(maxCount);
          float intensity = float(i) / 255.0f;
          tf_data[i] = glm::vec4(intensity, intensity, intensity, normalized);
      }
  }

// refine the TF with persistent homology data
void WorkContext::refine_with_ph(const Volume &volume, int ph_threshold, std::vector<glm::vec4> &tf_data)
{
  // compute raw persistence pairs
  std::vector<int> filt_vals;
  auto raw_pairs = calculate_persistence_pairs(volume, filt_vals);

  // apply the threshold cut to remove low-persistence features
  auto filtered = threshold_cut(raw_pairs, ph_threshold);

  // compute the maximum persistence value from the filtered pairs
  float maxPersistence = 0.0f;
  for (const auto &pair : filtered)
  {
      float persistence = (pair.death > pair.birth) ? float(pair.death - pair.birth) : 0.0f;
      if (persistence > maxPersistence)
          maxPersistence = persistence;
  }
  if (maxPersistence < 1e-6f)
      maxPersistence = 1.0f;

  // for each filtered pair, compute a weight and blend the highlight into the base TF
  for (auto &pair : filtered) 
  {
      float persistence = (pair.death > pair.birth) ? float(pair.death - pair.birth) : 0.0f;
      float weight = persistence / maxPersistence;
      weight = glm::clamp(weight, 0.2f, 1.0f);

      // convert birth and death to indices in the transfer function range [0,255]
      uint32_t b = std::clamp(pair.birth, static_cast<uint32_t>(0), static_cast<uint32_t>(255));
      uint32_t d = std::clamp(pair.death, static_cast<uint32_t>(0), static_cast<uint32_t>(255));

      // blend the base TF color with red for the birth index
      glm::vec4 baseBirth = tf_data[b];
      glm::vec4 redHighlight = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
      tf_data[b] = glm::mix(baseBirth, redHighlight, weight);

      // blend the base TF color with green for the death index
      glm::vec4 baseDeath = tf_data[d];
      glm::vec4 greenHighlight = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
      tf_data[d] = glm::mix(baseDeath, greenHighlight, weight);
  }
}

void WorkContext::load_persistence_diagram_texture(const std::string &filePath)
{
  try {
      persistence_texture_resource.construct(filePath);
      ui.set_persistence_texture(persistence_texture_resource.getImTextureID());
  } catch (const std::exception& e) {
      std::cerr << "Failed to load persistence diagram texture: " << e.what() << std::endl;
  }
}

void WorkContext::highlight_persistence_pair(const PersistencePair& pair)
{
    std::cout << "DEBUG: highlight_persistence_pair invoked with (birth=" 
              << pair.birth << ", death=" << pair.death << ")" << std::endl;
    std::vector<glm::vec4> tf_data;
    transfer_function.update(persistence_pairs, *ui.get_volume(), tf_data);
    auto [vol_min, vol_max] = transfer_function.compute_min_max_scalar(*ui.get_volume());
    
    float normalizedBirth = (float(pair.birth) - vol_min) / float(vol_max - vol_min);
    float normalizedDeath = (float(pair.death) - vol_min) / float(vol_max - vol_min);
    uint32_t indexBirth = static_cast<uint32_t>(normalizedBirth * 255.0f);
    uint32_t indexDeath = static_cast<uint32_t>(normalizedDeath * 255.0f);
    indexBirth = std::clamp(indexBirth, 0u, 255u);
    indexDeath = std::clamp(indexDeath, 0u, 255u);

    if (indexBirth == indexDeath) 
    {
        const uint32_t delta = 5;
        indexBirth = (indexBirth >= delta) ? indexBirth - delta : 0;
        indexDeath = std::min(indexDeath + delta, 255u);
    }
    
    for (uint32_t i = indexBirth; i <= indexDeath && i < tf_data.size(); ++i)
    {
        tf_data[i] = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);  // highlight in yellow
    }
    
    storage.get_buffer_by_name("transfer_function").update_data_bytes(tf_data.data(), sizeof(glm::vec4) * tf_data.size());
    vmc.logical_device.get().waitIdle();
    
    std::cout << "DEBUG: Highlighted persistence pair in range [" << indexBirth << ", " << indexDeath << "]." << std::endl;
}

void WorkContext::isolate_persistence_pairs(const std::vector<PersistencePair>& pairs)
{
    std::vector<glm::vec4> tf_data(256, glm::vec4(0.0f));

    auto [vol_min, vol_max] = transfer_function.compute_min_max_scalar(*ui.get_volume());
    auto normalize = [&](uint32_t v)
    {
        float t = float(v - vol_min) / float(vol_max - vol_min);
        return uint32_t(std::clamp(t * 255.0f, 0.0f, 255.0f));
    };

    // paint every bin in [birth..death] solid red
    glm::vec4 highlight(1.0f, 0.0f, 0.0f, 1.0f);
    for (auto &p : pairs)
    {
        uint32_t bi = normalize(p.birth);
        uint32_t di = normalize(p.death);
        if (bi > di) std::swap(bi, di);
        for (uint32_t i = bi; i <= di && i < tf_data.size(); ++i)
        {
            tf_data[i] = highlight;
        }
    }

    storage.get_buffer_by_name("transfer_function").update_data_bytes(tf_data.data(), sizeof(glm::vec4)*tf_data.size());
    vmc.logical_device.get().waitIdle();
}

void WorkContext::volume_highlight_persistence_pairs(const std::vector<PersistencePair>& pairs)
{
    std::vector<glm::vec4> tf_data(256, glm::vec4(0.0f));

    // paint every bin in [birth..death] with a low‐alpha red
    glm::vec4 highlight(1.0f, 0.0f, 0.0f, 1.0f);
    for (auto &p : pairs)
    {
        uint32_t b = std::clamp(p.birth, 0u, 255u);
        uint32_t d = std::clamp(p.death, 0u, 255u);
        for (uint32_t i = b; i <= d; ++i)
            tf_data[i] = highlight;
    }

    storage.get_buffer_by_name("transfer_function").update_data_bytes(tf_data.data(), sizeof(glm::vec4) * tf_data.size());
    vmc.logical_device.get().waitIdle();
}

void WorkContext::set_raw_persistence_pairs(const std::vector<PersistencePair>& pairs)
{
    raw_persistence_pairs = pairs;
    // tell the UI to use these for drawing
    ui.set_persistence_pairs(&raw_persistence_pairs);
}
} // namespace ve