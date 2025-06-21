#include "work_context.hpp"
#include <vulkan/vulkan_enums.hpp>
#include "stb/stb_image.h"
#include "transfer_function.hpp"
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

namespace ve
{
WorkContext::WorkContext(const VulkanMainContext& vmc, VulkanCommandContext& vcc) : vmc(vmc), vcc(vcc), storage(vmc, vcc), swapchain(vmc, vcc, storage), renderer(vmc, storage), ray_marcher(vmc, storage), persistence_texture_resource(vmc, storage), ui(vmc) {}

void WorkContext::construct(AppState& app_state, const Volume& volume)
{
  vcc.add_graphics_buffers(frames_in_flight);
  vcc.add_compute_buffers(2);
  vcc.add_transfer_buffers(1);
  renderer.setup_storage(app_state);
  gradient_volume = compute_gradient_volume(volume);
  ray_marcher.setup_storage(app_state, volume, gradient_volume);
  app_state.max_gradient = *std::max_element(gradient_volume.data.cbegin(), gradient_volume.data.cend());
  swapchain.construct(app_state.vsync);
  app_state.set_window_extent(swapchain.get_extent());
  for (uint32_t i = 0; i < frames_in_flight; ++i)
  {
    syncs.emplace_back(vmc.logical_device.get());
    device_timers.emplace_back(vmc);
  }
  renderer.construct(swapchain.get_render_pass(), app_state);
  ray_marcher.construct(app_state, vcc, volume.resolution);
  ui.construct(vcc, swapchain.get_render_pass(), frames_in_flight);
  ui.set_transfer_function(&transfer_function);
  scalar_volume = &volume;
  ui.set_volume(scalar_volume);

  // compute and set scalar persistence pairs
  std::vector<int> filt_vals;
  persistence_pairs = calculate_persistence_pairs(volume, filt_vals, app_state.filtration_mode);
  ui.set_persistence_pairs(&persistence_pairs);
  set_persistence_pairs(persistence_pairs, volume);

  // compute and set gradient persistence pairs
  
  std::vector<int> grad_filt_vals;
  auto raw_grad_pairs = calculate_persistence_pairs(gradient_volume, grad_filt_vals, app_state.filtration_mode);
  gradient_persistence_pairs.clear();
  for (auto &p : raw_grad_pairs)
  {
    uint32_t b = grad_filt_vals[p.birth];
    uint32_t d = grad_filt_vals[p.death];
    gradient_persistence_pairs.emplace_back(b, d);
  }
  ui.set_gradient_persistence_pairs(&gradient_persistence_pairs);

  merge_tree = build_merge_tree_with_tolerance(persistence_pairs, 5u);
  ui.set_merge_tree(&merge_tree);

  ui.set_gradient_volume(&gradient_volume);

  // switching between scalar/gradient persistenceColor Ramp
  ui.set_on_merge_mode_changed([this](int mode)
  {
    if (mode == 0)
    {
      // scalar mode
      ui.set_persistence_pairs(&persistence_pairs);
      ui.set_gradient_persistence_pairs(nullptr);

      if (scalar_volume && !persistence_pairs.empty())
      {
        set_persistence_pairs(persistence_pairs, *scalar_volume);
      }
    }
    else
    {
      // gradient mode
      ui.set_persistence_pairs(nullptr);
      ui.set_gradient_persistence_pairs(&gradient_persistence_pairs);

      if (&gradient_volume && !gradient_persistence_pairs.empty())
      {
        transfer_function.update(gradient_persistence_pairs, gradient_volume, tf_data);
      }
    }
    merge_tree = build_merge_tree_with_tolerance((mode == 0 ? persistence_pairs : gradient_persistence_pairs), 5u);
    ui.mark_merge_tree_dirty();
    ui.clear_selection();
    ui.mark_tf2d_hist_dirty();
  });

  ui.set_on_highlight_selected([this](const std::vector<std::pair<PersistencePair,float>>& hits, int ramp_index)
  {
    this->volume_highlight_persistence_pairs_gradient(hits, ramp_index);
  });

  ui.set_on_diff_selected([this](const PersistencePair &a, const PersistencePair &b) {
    this->highlight_diff(a,b);
  });

  ui.set_on_intersect_selected([this](const PersistencePair &a, const PersistencePair &b) {
    this->highlight_intersection(a, b);
  });
  ui.set_on_union_selected([this](const PersistencePair &a, const PersistencePair &b) {
      this->highlight_union(a, b);
  });

  ui.set_on_custom_color_chosen([this](const std::vector<PersistencePair>& pairs, const ImVec4& color)
  {
    this->apply_custom_color_to_volume(pairs, color);
  });

  ui.set_on_clear_custom_colors([this]()
  {
    this->reset_custom_colors();
  });

  export_persistence_pairs_to_csv(persistence_pairs, gradient_persistence_pairs, "scalar_pairs.csv", "gradient_pairs.csv");

  // load static persistence diagram texture (for reference)
  load_persistence_diagram_texture("output_plots/persistence_diagram.png");
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
  syncs[0].wait_for_fence(Synchronization::F_RENDER_FINISHED);
  syncs[0].reset_fence(Synchronization::F_RENDER_FINISHED);
  if (app_state.total_frames > frames_in_flight)
  {
    // update device timers
    for (int i = 0; i < DeviceTimer::TIMER_COUNT; i++) app_state.device_timings[i] = device_timers[0].get_result_by_idx(i);
  }

  auto &buf = storage.get_buffer_by_name("transfer_function");
  buf.update_data(tf_data);
  vmc.logical_device.get().waitIdle();

  vk::ResultValue<uint32_t> image_idx = vmc.logical_device.get().acquireNextImageKHR(swapchain.get(), uint64_t(-1), syncs[0].get_semaphore(Synchronization::S_IMAGE_AVAILABLE));
  VE_CHECK(image_idx.result, "Failed to acquire next image!");

  uint32_t read_only_image = (app_state.total_frames / frames_in_flight) % frames_in_flight;
  if (app_state.save_screenshot)
  {
    storage.get_image_by_name("ray_marcher_output_texture").save_to_file(vcc);
    app_state.save_screenshot = false;
  }
  render(image_idx.value, app_state, read_only_image);
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
  syncs[0].wait_for_fence(Synchronization::F_COMPUTE_FINISHED);
  // reset fence for the next compute iteration
  syncs[0].reset_fence(Synchronization::F_COMPUTE_FINISHED);

  app_state.cam.update_data();
  storage.get_buffer_by_name("ray_marcher_uniform_buffer").update_data_bytes(&app_state.cam.data, sizeof(Camera::Data));

  vk::CommandBuffer &cb = vcc.get_one_time_transfer_buffer();

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

  vk::CommandBuffer &compute_cb = vcc.begin(vcc.compute_cbs[0]);
  device_timers[0].reset(compute_cb, {DeviceTimer::VOLUME});
  device_timers[0].start(compute_cb, DeviceTimer::VOLUME, vk::PipelineStageFlagBits::eComputeShader);
  ray_marcher.compute(compute_cb, app_state, read_only_buffer_idx);
  device_timers[0].stop(compute_cb, DeviceTimer::VOLUME, vk::PipelineStageFlagBits::eComputeShader);
  compute_cb.end();
  read_only_buffer_idx = (read_only_buffer_idx + 1) % frames_in_flight;

  vk::SubmitInfo compute_si(0, nullptr, nullptr, 1, &vcc.compute_cbs[0]);
  vmc.get_compute_queue().submit(compute_si, syncs[0].get_fence(Synchronization::F_COMPUTE_FINISHED));

  vk::CommandBuffer& graphics_cb = vcc.begin(vcc.graphics_cbs[0]);
  device_timers[0].reset(graphics_cb, {DeviceTimer::UI});
  if (render_texture.get_layout() != vk::ImageLayout::eShaderReadOnlyOptimal)
  {
    perform_image_layout_transition(graphics_cb, render_texture.get_image(), render_texture.get_layout(), vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eMemoryRead, 0, 1, 1);
    render_texture.set_layout(vk::ImageLayout::eShaderReadOnlyOptimal);
  }
  renderer.render(graphics_cb, app_state, read_only_buffer_idx, swapchain.get_framebuffer(image_idx), swapchain.get_render_pass().get());
  device_timers[0].start(graphics_cb, DeviceTimer::UI, vk::PipelineStageFlagBits::eTopOfPipe);
  if (app_state.show_ui) ui.draw(graphics_cb, app_state);
  device_timers[0].stop(graphics_cb, DeviceTimer::UI, vk::PipelineStageFlagBits::eBottomOfPipe);
  graphics_cb.endRenderPass();
  graphics_cb.end();

  std::vector<vk::Semaphore> render_wait_semaphores;
  std::vector<vk::PipelineStageFlags> render_wait_stages;
  render_wait_semaphores.push_back(syncs[0].get_semaphore(Synchronization::S_IMAGE_AVAILABLE));
  render_wait_stages.push_back(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  std::vector<vk::Semaphore> render_signal_semaphores;
  render_signal_semaphores.push_back(syncs[0].get_semaphore(Synchronization::S_RENDER_FINISHED));
  vk::SubmitInfo render_si(render_wait_semaphores.size(), render_wait_semaphores.data(), render_wait_stages.data(), 1, &vcc.graphics_cbs[0], render_signal_semaphores.size(), render_signal_semaphores.data());
  vmc.get_graphics_queue().submit(render_si, syncs[0].get_fence(Synchronization::F_RENDER_FINISHED));

  vk::PresentInfoKHR present_info(1, &syncs[0].get_semaphore(Synchronization::S_RENDER_FINISHED), 1, &swapchain.get(), &image_idx);
  VE_CHECK(vmc.get_present_queue().presentKHR(present_info), "Failed to present image!");
}

void WorkContext::set_persistence_pairs(const std::vector<PersistencePair>& pairs, const Volume& volume)
{
  persistence_pairs = pairs;

  // compute the global max persistence, later used in isolate/volumeHighlight
  global_max_persistence = 1;
  for (auto &p : persistence_pairs)
  {
    uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
    global_max_persistence = std::max(global_max_persistence, pers);
  }

  transfer_function.update(persistence_pairs, volume, tf_data);
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

void WorkContext::volume_highlight_persistence_pairs_gradient(const std::vector<std::pair<PersistencePair, float>>& pairs, int ramp_index)
{
  tf_data.assign(256, glm::vec4(0.0f));

  uint32_t max_pers = std::max(global_max_persistence, 1u);
  for (auto& entry : pairs)
  {
    const auto& p = entry.first;
    float brush_op = entry.second;

    uint32_t pers = (p.death > p.birth ? (p.death - p.birth) : 0);
    float tColor = float(pers) / float(max_pers);
    glm::vec3 rgb;
    switch (ramp_index)
    {
      case UI::RAMP_HSV:
      {
        float hue = (1.0f - tColor) * 0.66f;
        ImGui::ColorConvertHSVtoRGB(hue, 1.0, 1.0, rgb.x, rgb.y, rgb.z);
      } break;
      case UI::RAMP_VIRIDIS:
        rgb = viridis(tColor);
        break;
      case UI::RAMP_PLASMA:
        rgb = plasma(tColor);
        break;
      case UI::RAMP_MAGMA:
        rgb = magma(tColor);
        break;
      case UI::RAMP_INFERNO:
        rgb = inferno(tColor);
        break;
      case UI::RAMP_CUSTOM:
      {
        ImVec4 sc = ui.get_custom_start_color();
        ImVec4 ec = ui.get_custom_end_color();
        glm::vec3 c0{sc.x, sc.y, sc.z};
        glm::vec3 c1{ec.x, ec.y, ec.z};
        rgb = glm::mix(c0, c1, tColor);
      } break;
      default:
      {
        float hue = (1.0f - tColor) * 0.66f;
        ImGui::ColorConvertHSVtoRGB(hue, 1.0, 1.0, rgb.x, rgb.y, rgb.z);
      }
    }

    float alpha = brush_op;
    if (ramp_index == UI::RAMP_CUSTOM)
    {
      alpha *= ui.get_custom_falloff();
    }

    uint32_t bi = std::clamp(p.birth, 0u, 255u);
    uint32_t di = std::clamp(p.death, 0u, 255u);
    if (bi > di) std::swap(bi, di);

    for (uint32_t i = bi; i <= di; ++i)
    {
      tf_data[i] = glm::vec4(rgb, alpha);
    }
  }

  for (auto& assign : custom_colors)
  {
    auto& p  = assign.first;
    auto& col = assign.second;
    uint32_t b = std::clamp(p.birth, 0u, 255u);
    uint32_t d = std::clamp(p.death, 0u, 255u);
    if (b > d) std::swap(b, d);

    for (uint32_t i = b; i <= d; ++i)
    {
      tf_data[i] = col;
    }
  }
}

static std::pair<uint32_t, uint32_t> clamp_and_sort_range(const PersistencePair& p)
{
  constexpr uint32_t maxBin = AppState::TF2D_BINS - 1u;
  uint32_t low = (p.birth < AppState::TF2D_BINS ? p.birth : maxBin);
  uint32_t high = (p.death < AppState::TF2D_BINS ? p.death : maxBin);
  if (low > high) std::swap(low, high);
  return { low, high };
}

void WorkContext::highlight_diff(const PersistencePair &base, const PersistencePair &mask)
{
  tf_data.assign(AppState::TF2D_BINS, glm::vec4(0.0f));

  std::pair<uint32_t,uint32_t> rangeA = clamp_and_sort_range(base);
  uint32_t b0 = rangeA.first;
  uint32_t d0 = rangeA.second;

  std::pair<uint32_t,uint32_t> rangeB = clamp_and_sort_range(mask);
  uint32_t b1 = rangeB.first;
  uint32_t d1 = rangeB.second;

  // fill [b0..d0] with diff_color (if enabled)
  if (ui.diff_enabled)
  {
    ImVec4 c = ui.diff_color;
    glm::vec3 diff_col{c.x, c.y, c.z};
    float alpha = c.w;

    for (uint32_t i = b0; i <= d0; ++i)
    {
      tf_data[i] = glm::vec4(diff_col, alpha);
    }
  }

  // mask out [b1..d1] (transparent)
  for (uint32_t i = b1; i <= d1; ++i)
  {
    tf_data[i] = glm::vec4(0.0f);
  }
}

void WorkContext::highlight_intersection(const PersistencePair& a, const PersistencePair& b)
{
  tf_data.assign(AppState::TF2D_BINS, glm::vec4(0.0f));

  std::pair<uint32_t,uint32_t> rangeA = clamp_and_sort_range(a);
  uint32_t a0 = rangeA.first;
  uint32_t a1 = rangeA.second;

  std::pair<uint32_t,uint32_t> rangeB = clamp_and_sort_range(b);
  uint32_t b0 = rangeB.first;
  uint32_t b1 = rangeB.second;

  // intersection bounds
  uint32_t start = std::max(a0, b0);
  uint32_t end   = std::min(a1, b1);

  // paint common intersection
  if (ui.intersect_enabled_common && start <= end)
  {
    ImVec4 c = ui.intersect_color_common;
    glm::vec4 col{c.x, c.y, c.z, c.w};
    for (uint32_t i = start; i <= end; ++i)
    {
      tf_data[i] = col;
    }
}

  // paint A-only before/after
  if (ui.intersect_enabled_Aonly)
  {
    ImVec4 cA = ui.intersect_color_Aonly;
    glm::vec4 colA{cA.x, cA.y, cA.z, cA.w};
    for (uint32_t i = a0; i < start; ++i)
    {
      tf_data[i] = colA;
    }
    for (uint32_t i = end + 1; i <= a1; ++i)
    {
      tf_data[i] = colA;
    }
  }

  // paint B-only before/after
  if (ui.intersect_enabled_Bonly)
  {
    ImVec4 cB = ui.intersect_color_Bonly;
    glm::vec4 colB{cB.x, cB.y, cB.z, cB.w};
    for (uint32_t i = b0; i < start; ++i)
    {
      tf_data[i] = colB;
    }
    for (uint32_t i = end + 1; i <= b1; ++i)
    {
      tf_data[i] = colB;
    }
  }
}
void WorkContext::highlight_union(const PersistencePair& a, const PersistencePair& b)
{
  tf_data.assign(AppState::TF2D_BINS, glm::vec4(0.0f));

  std::pair<uint32_t,uint32_t> rangeA = clamp_and_sort_range(a);
  uint32_t a0 = rangeA.first;
  uint32_t a1 = rangeA.second;

  std::pair<uint32_t,uint32_t> rangeB = clamp_and_sort_range(b);
  uint32_t b0 = rangeB.first;
  uint32_t b1 = rangeB.second;

  // paint A range if enabled
  if (ui.union_enabled_Aonly)
  {
    ImVec4 cA = ui.union_color_Aonly;
    glm::vec4 colA{cA.x, cA.y, cA.z, cA.w};
    for (uint32_t i = a0; i <= a1; ++i)
    {
      tf_data[i] = colA;
    }
  }

  // paint common or B-only
  for (uint32_t i = b0; i <= b1; ++i)
  {
    bool inA = (i >= a0 && i <= a1);
    if (inA)
    {
      if (ui.union_enabled_common)
      {
        ImVec4 cC = ui.union_color_common;
        tf_data[i] = glm::vec4(cC.x, cC.y, cC.z, cC.w);
      }
    }
    else 
    {
      if (ui.union_enabled_Bonly)
      {
        ImVec4 cB = ui.union_color_Bonly;
        tf_data[i] = glm::vec4(cB.x, cB.y, cB.z, cB.w);
      }
    }
  }
}

void WorkContext::set_raw_persistence_pairs(const std::vector<PersistencePair>& pairs)
{
  raw_persistence_pairs = pairs;
  ui.set_persistence_pairs(&raw_persistence_pairs);
}

void WorkContext::set_gradient_persistence_pairs(const std::vector<PersistencePair>& pairs)
{
  gradient_persistence_pairs = pairs;
  ui.set_gradient_persistence_pairs(&gradient_persistence_pairs);
}

void WorkContext::export_persistence_pairs_to_csv(const std::vector<PersistencePair>& scalar_pairs, const std::vector<PersistencePair>& gradient_pairs, const std::string& scalar_filename, const std::string& gradient_filename) const
{
  if (mkdir("volume_data", 0755) != 0 && errno != EEXIST)
  {
    std::cerr << "Error: could not create directory 'volume_data'\n";
    return;
  }

  std::string scalar_path   = std::string("volume_data/") + scalar_filename;
  std::string gradient_path = std::string("volume_data/") + gradient_filename;

  // write scalar-mode pairs to CSV
  std::ofstream out_scalar(scalar_path);
  if (!out_scalar)
  {
      std::cerr << "Error: could not open '" << scalar_filename << "' for writing\n";
      return;
  }
  out_scalar << "birth,death\n";
  for (const auto& p : scalar_pairs)
  {
      out_scalar << p.birth << "," << p.death << "\n";
  }

  // write gradient-mode pairs to CSV
  std::ofstream out_grad(gradient_path);
  if (!out_grad)
  {
      std::cerr << "Error: could not open '" << gradient_filename << "' for writing\n";
      return;
  }
  out_grad << "birth,death\n";
  for (const auto& p : gradient_pairs)
  {
      out_grad << p.birth << "," << p.death << "\n";
  }

  std::cout << "Exported persistence pairs to:\n" << "  - " << scalar_filename  << "\n" << "  - " << gradient_filename << "\n";
}

void WorkContext::apply_custom_color_to_volume(const std::vector<PersistencePair>& pairs, const ImVec4& color)
{
  glm::vec4 chosen_color(color.x, color.y, color.z, color.w);

  // record every new assignment
  for (auto &p : pairs)
  {
      custom_colors.emplace_back(p, chosen_color);
  }

  // replay *all* custom assignments
  for (auto &assign : custom_colors)
  {
    const auto &p = assign.first;
    const auto &col = assign.second;
    uint32_t b = std::clamp(p.birth, uint32_t(0), uint32_t(255));
    uint32_t d = std::clamp(p.death, uint32_t(0), uint32_t(255));
    if (b > d) std::swap(b, d);
    for (uint32_t i = b; i <= d; ++i)
    {
        tf_data[i] = col;
    }
  }
}

void WorkContext::reset_custom_colors()
{
  ui.clear_selection();
  custom_colors.clear();

  int ramp = ui.get_selected_ramp();

  std::vector<std::pair<PersistencePair, float>> all_hits;
  all_hits.reserve(persistence_pairs.size());
  for (const auto &p : persistence_pairs)
      all_hits.emplace_back(p, 1.0f);

  volume_highlight_persistence_pairs_gradient(all_hits, ramp);
}

}//namespace ve
