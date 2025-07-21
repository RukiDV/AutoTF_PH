#include "work_context.hpp"
#include <vulkan/vulkan_enums.hpp>
#include "stb/stb_image.h"
#include "transfer_function.hpp"
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include "stb/stb_image_write.h"
#include <iomanip>

namespace ve
{
WorkContext::WorkContext(const VulkanMainContext& vmc, VulkanCommandContext& vcc, std::vector<PersistencePair> raw_pairs, std::vector<int> raw_filt, std::vector<PersistencePair> raw_grad_pairs, std::vector<int> raw_grad_filt) : vmc(vmc), vcc(vcc), raw_persistence_pairs(std::move(raw_pairs)), scalar_filtration(std::move(raw_filt)), raw_gradient_pairs(std::move(raw_grad_pairs)), gradient_filtration(std::move(raw_grad_filt)), storage(vmc, vcc), swapchain(vmc, vcc, storage), renderer(vmc, storage), ray_marcher(vmc, storage), persistence_texture_resource(vmc, storage), ui(vmc) {}

void WorkContext::fillTF2DFromVolume(const Volume& vol)
{
  int B = AppState::TF2D_BINS;
  tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));

  bool gradMode = (ui.get_pd_mode()==1);
  for (size_t i = 0; i < vol.data.size(); ++i)
  {
    int s = int(vol.data[i]);
    int g = int(gradient_volume.data[i]);
    if (s<0||s>=B||g<0||g>=B) continue;

    int col, row;
    if (!gradMode)
    {
      // scalar mode
      col = s;
      row = (B-1) - g;
    }
    else
    {
      // gradient mode
      col =  g;
      row = (B-1) - s;
    }
    tf_data[row * B + col] = glm::vec4(1,1,1,1);
  }
}

void WorkContext::construct(AppState& app_state, const Volume& volume)
{
  vcc.add_graphics_buffers(frames_in_flight);
  vcc.add_compute_buffers(2);
  vcc.add_transfer_buffers(1);
  renderer.setup_storage(app_state);
  gradient_volume = compute_gradient_volume(volume);
  grads_by_scalar.clear();
  grads_by_scalar.resize(AppState::TF2D_BINS);
  for (size_t vid = 0; vid < volume.data.size(); ++vid)
  {
    int s = int(volume.data[vid]);
    int g = int(gradient_volume.data[vid]);
    if (s >= 0 && s < AppState::TF2D_BINS && g >= 0 && g < AppState::TF2D_BINS)
    {
      grads_by_scalar[s].push_back(g);
    }
  }
  ray_marcher.setup_storage(app_state, volume, gradient_volume);
  app_state.max_gradient = *std::max_element(gradient_volume.data.cbegin(), gradient_volume.data.cend());
  swapchain.construct(false);
  app_state.set_window_extent(swapchain.get_extent());
  for (uint32_t i = 0; i < frames_in_flight; ++i)
  {
    syncs.emplace_back(vmc.logical_device.get());
    device_timers.emplace_back(vmc);
  }
  renderer.construct(swapchain.get_render_pass(), app_state);
  ray_marcher.construct(app_state, vcc, volume.resolution);
  ui.construct(vcc, swapchain.get_render_pass(), frames_in_flight);
  auto t12 = timer.restart<ms>();
  std::cout << "[TIMING] UI construct: " << t12 << " ms\n";

  ui.set_transfer_function(&transfer_function);
  auto t11 = timer.restart<ms>();
  std::cout << "[TIMING] set_transfer_fucntion_UI: " << t11 << " ms\n";

  scalar_volume = &volume;
  ui.set_volume(scalar_volume);
  auto t10 = timer.restart<ms>();
    std::cout << "[TIMING] set_volume_UI: " << t10 << " ms\n";

  // compute and set scalar persistence pairs
  std::vector<int> filt_vals;
  //persistence_pairs = calculate_persistence_pairs(volume, filt_vals, app_state.filtration_mode);
  persistence_pairs.clear();
  for (auto &p : raw_persistence_pairs)
  {
    uint32_t b = scalar_filtration[p.birth];
    uint32_t d = scalar_filtration[p.death];
    persistence_pairs.emplace_back(b, d);
  }
  auto t1 = timer.restart<ms>();
  std::cout << "[TIMING] calculate_persistence_pairs_scalar: " << t1 << " ms\n";

  ui.set_persistence_pairs(&persistence_pairs);
  auto t2 = timer.restart<ms>();
    std::cout << "[TIMING] set_scalar_persistence_pairs_UI: " << t2 << " ms\n";

  set_persistence_pairs(persistence_pairs, volume);
  auto t3 = timer.restart<ms>();
  std::cout << "[TIMING] set_persistence_pairs: " << t3 << " ms\n";

  // compute and set gradient persistence pairs
  
  std::vector<int> grad_filt_vals;
  //auto raw_grad_pairs = calculate_persistence_pairs(gradient_volume, grad_filt_vals, app_state.filtration_mode);
  auto t4 = timer.restart<ms>();
  std::cout << "[TIMING] calculate_persistence_pairs_gradient: " << t4 << " ms\n";

  gradient_persistence_pairs.clear();
  for (auto &p : raw_gradient_pairs)
  {
    uint32_t b = gradient_filtration[p.birth];
    uint32_t d = gradient_filtration[p.death];
    gradient_persistence_pairs.emplace_back(b, d);
  }
  ui.set_gradient_persistence_pairs(&gradient_persistence_pairs);
  auto t5 = timer.restart<ms>();
  std::cout << "[TIMING] set_gradient_persistence_pairs_UI: " << t5 << " ms\n";

  merge_tree = build_merge_tree_with_tolerance(persistence_pairs, 5u);
  ui.set_merge_tree(&merge_tree);

  ui.set_gradient_volume(&gradient_volume);

  tf_data.clear();
  tf_data.resize(AppState::TF2D_BINS * AppState::TF2D_BINS);
  for (int i = 0; i < AppState::TF2D_BINS; ++i) 
  {
    for (int j = 0; j < AppState::TF2D_BINS; ++j)
    {
      float value = float(j) / float(AppState::TF2D_BINS - 1);
      tf_data[j * AppState::TF2D_BINS + i] = glm::vec4(1.0, value, value, 1.0);
    }
  }
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

      global_max_persistence = 1;
      for (auto &p : gradient_persistence_pairs)
      {
        uint32_t pers = (p.death > p.birth ? (p.death - p.birth) : 0);
        global_max_persistence = std::max(global_max_persistence, pers);
      }
       if (&gradient_volume && !gradient_persistence_pairs.empty())
      {
        transfer_function.update(gradient_persistence_pairs, gradient_volume, tf_data);
      }
    }
    merge_tree = build_merge_tree_with_tolerance((mode == 0 ? persistence_pairs : gradient_persistence_pairs), 5u);
    ui.mark_merge_tree_dirty();
    ui.clear_selection();
  });

  ui.set_on_highlight_selected([this](const std::vector<std::pair<PersistencePair,float>>& hits, int ramp_index)
  {
    this->volume_highlight_persistence_pairs(hits, ramp_index);
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

  ui.set_on_onlyA_selected([this](const PersistencePair& a, const PersistencePair& b, const ImVec4& col){
    this->highlight_onlyA(a, b, col);
  });
  ui.set_on_onlyB_selected([this](const PersistencePair& a, const PersistencePair& b, const ImVec4& col){
      this->highlight_onlyB(a, b, col);
  });

  ui.set_on_custom_color_chosen([this](const std::vector<PersistencePair>& pairs, const ImVec4& color)
  {
    this->apply_custom_color_to_volume(pairs, color);
  });

  ui.set_on_clear_custom_colors([this]()
  {
    this->reset_custom_colors();
  });

  ui.set_on_tf2d_selected([this](auto const& bins, ImVec4 col)
  {
    bool gradMode = (ui.get_pd_mode() == 1);

    tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));

    for (auto& b : bins)
    {
      int x = gradMode ? b.second : b.first;
      int y = gradMode ? b.first  : b.second;
      if (gradMode) 
        y = (AppState::TF2D_BINS - 1) - y;

      tf_data[y * AppState::TF2D_BINS + x] = glm::vec4(col.x, col.y, col.z, col.w);
    }

    ui.persistence_bins = click_bins;
    ui.persistence_bin_colors = click_colors;

    std::unordered_set<uint32_t> drawn;
    drawn.reserve(bins.size());
    for (auto& b : bins)
    {
      int x = b.first, y = b.second;
      if (gradMode)
      {
        std::swap(x,y);
        y = (AppState::TF2D_BINS - 1) - y;
      }
      uint32_t key = (uint32_t(x)<<16) | uint32_t(y);
      drawn.insert(key);
    }

    ImU32 ucol = ImGui::ColorConvertFloat4ToU32(col);
    ui.persistence_bins.clear();
    ui.persistence_bin_colors.clear();
    for (size_t i = 0; i < scalar_volume->data.size(); ++i)
    {
      int s = int(scalar_volume->data[i]);
      int g = int(gradient_volume.data[i]);
      if (s < 0 || s >= AppState::TF2D_BINS || g < 0 || g >= AppState::TF2D_BINS) continue;

      int x = gradMode ? g : s;
      int y = gradMode ? s : g;
      if (gradMode) 
        y = (AppState::TF2D_BINS - 1) - y;

      uint32_t key = (uint32_t(x)<<16) | uint32_t(y);
      if (!drawn.count(key)) 
        continue;

      ui.persistence_bins.emplace_back(x, y);
      ui.persistence_bin_colors.push_back(ucol);
    }
    last_tf2d_bins = ui.persistence_bins;
  });


  ui.set_on_reproject([this]() 
  { 
    reproject_and_compare(); 
  });
  
  ui.set_on_persistence_reprojected([this](int featIdx)
  {
    std::cout << "[WC] on_persistence_reprojected called for featIdx=" << featIdx << "\n";
    pending_reproject_idx = featIdx;

    ui.persistence_bins.clear();
    ui.persistence_bin_colors.clear();
    ImU32 green = IM_COL32(0,255,0,200);

    bool gradMode = (ui.get_pd_mode() == 1);
    const auto &pairs = gradMode ? gradient_persistence_pairs : persistence_pairs;

    auto p = pairs[featIdx];
    int b = std::clamp<int>(p.birth, 0, AppState::TF2D_BINS-1);
    int d = std::clamp<int>(p.death, 0, AppState::TF2D_BINS-1);
    if (b > d) std::swap(b,d);
    if (b == d)
    {
      b = std::max(b-1, 0); d = std::min(d+1.0, AppState::TF2D_BINS - 1.0);
    }

    // primary (birth‒death) and secondary (gradient) clamps
    int s_min, s_max, g_min, g_max;

    if (!gradMode)
    {
      s_min = b;
      s_max = d;
      g_min = 0;
      g_max = AppState::TF2D_BINS - 1;
    } else
    {
      /*s_min = b;
      s_max = d;
      g_min = 0;
      g_max = B-1;*/
      g_min = b;
      g_max = d;
      s_min = 0;
      s_max = AppState::TF2D_BINS - 1;
    }
    // secondary slider (top/bottom)
    if (ui.tf2d_use_secondary)
    {
      int fmin = ui.tf2d_secondary_min;
      int fmax = ui.tf2d_secondary_max;
      if (!gradMode)
      {
        g_min = (AppState::TF2D_BINS - 1) - fmax;
        g_max = (AppState::TF2D_BINS - 1) - fmin;
      } else
      {
       s_min = (AppState::TF2D_BINS - 1) - fmax;
        s_max = (AppState::TF2D_BINS - 1) - fmin; 
      }
    }

    // primary slider (left/right)
    if (ui.tf2d_use_primary)
    {
      if (!gradMode)
      {
        s_min = ui.tf2d_primary_min;
        s_max = ui.tf2d_primary_max;
      } else
      {
        int fmin = ui.tf2d_primary_min;
        int fmax = ui.tf2d_primary_max;
        g_min = (AppState::TF2D_BINS - 1) - fmax;
        g_max = (AppState::TF2D_BINS - 1) - fmin;
      }
    }


    // gather voxels that pass both clamps
    for (size_t i = 0; i < scalar_volume->data.size(); ++i)
    {
        int s = int(scalar_volume->data[i]);
        int g = int(gradient_volume.data[i]);
        if (s<0||s>= AppState::TF2D_BINS||g<0||g>=AppState::TF2D_BINS) continue;
        if (s < s_min || s > s_max) continue;
        if (g < g_min || g > g_max) continue;

        int fg = (AppState::TF2D_BINS - 1) - g;

        int x, y;
        if (!gradMode)
        {
            // scalar mode
            x = s;
            y = fg;
        } else {
            // gradient mode
            x = (AppState::TF2D_BINS - 1) - fg;
            y = s;
            y = (AppState::TF2D_BINS - 1) - y;
        }

        ui.persistence_bins.emplace_back(x, y);
        ui.persistence_bin_colors.push_back(green);
    }

    click_bins = ui.persistence_bins;
    click_colors = ui.persistence_bin_colors;
    last_tf2d_bins = ui.persistence_bins;

    PersistencePair per = (gradMode ? gradient_persistence_pairs[featIdx] : persistence_pairs[featIdx]);
    std::vector<std::pair<PersistencePair,float>> single{{per, 1.0f}};
    this->volume_highlight_persistence_pairs(single, ui.get_selected_ramp());
  });

  ui.set_on_persistence_multi_reprojected([this](const std::vector<int>& featIdxs)
  {
    ui.persistence_bins.clear();
    ui.persistence_bin_colors.clear();

    bool gradMode = (ui.get_pd_mode() == 1);

    std::unordered_set<uint32_t> seen;
    seen.reserve(size_t(AppState::TF2D_BINS) * size_t(AppState::TF2D_BINS));
    
    // build both the 2D overlay *and* the list of pairs for volume‐highlight
    std::vector<std::pair<PersistencePair,float>> forVolume;
    forVolume.reserve(featIdxs.size());
    
    for (size_t fi = 0; fi < featIdxs.size(); ++fi)
    {
      int idx = featIdxs[fi];
      // record this pair for the 3D volume
      const auto& pairs = gradMode ? gradient_persistence_pairs : persistence_pairs;
      PersistencePair p = pairs[idx];
      forVolume.emplace_back(p, 1.0f);
      
      // color for this feature
      float hue = float(fi) / float(featIdxs.size());
      float cr, cg, cb;
      ImGui::ColorConvertHSVtoRGB(hue, 1, 1, cr, cg, cb);
      ImU32 colU = ImGui::ColorConvertFloat4ToU32({cr, cg, cb, 0.6f});
      
      // get birth/death clamped
      int bs = std::clamp<int>(p.birth,  0, AppState::TF2D_BINS - 1);
      int ds = std::clamp<int>(p.death,  0, AppState::TF2D_BINS - 1);
          if (bs > ds) std::swap(bs, ds);
          if (bs == ds) {
              bs = std::max(0, bs - 1);
              ds = std::min(int(AppState::TF2D_BINS - 1), ds+1);
          }

          int s0, s1, g0, g1;
          if (!gradMode)
          {
            s0 = bs; s1 = ds; g0 = 0;   g1 = AppState::TF2D_BINS - 1;
          }
          else
          {
            s0 = bs; s1 = ds; g0 = 0;   g1 = AppState::TF2D_BINS - 1;
          }

          // apply *this* feature's primary clamp
          if (ui.tf2d_use_primary)
          {
              auto pr = ui.primary_clamp_per_point[idx];
              if (!gradMode) {
                  s0 = pr[0]; s1 = pr[1];
              } else {
                  int fmin=pr[0], fmax=pr[1];
                  g0 = (AppState::TF2D_BINS - 1)-fmax; g1 = (AppState::TF2D_BINS - 1)-fmin;
              }
          }
          // apply *this* feature's secondary clamp
          if (ui.tf2d_use_secondary)
          {
              auto sr = ui.secondary_clamp_per_point[idx];
              if (!gradMode)
              {
                  int fmin=sr[0], fmax=sr[1];
                  g0 = (AppState::TF2D_BINS - 1)-fmax; g1 = (AppState::TF2D_BINS - 1)-fmin;
              } else {
                  s0 = sr[0]; s1 = sr[1];
              }
          }

          ui.persistence_bins.emplace_back( s0, (AppState::TF2D_BINS - 1) - g0 );
          ui.persistence_bin_colors.push_back(colU);

          ui.persistence_bins.emplace_back( s0 + 1, (AppState::TF2D_BINS - 1) - g0 + 1 );
          ui.persistence_bin_colors.push_back(colU);
          // collect 2D bins
          for (int s = s0; s <= s1; ++s)
          {
              for (int gval : grads_by_scalar[s])
              {
                  if (gval < g0 || gval > g1) continue;
                  int fg = (AppState::TF2D_BINS - 1) - gval;
                  uint32_t key = (uint32_t(s)<<16) | uint32_t(fg);
                  if (seen.insert(key).second) {
                      ui.persistence_bins.emplace_back(s, fg);
                      ui.persistence_bin_colors.push_back(colU);
                  }
              }
          }
      }

      last_tf2d_bins = ui.persistence_bins;
      int ramp = ui.get_selected_ramp();
      this->volume_highlight_persistence_pairs(forVolume, ramp);
  });
  ui.set_on_brush_selected([this](const std::vector<PersistencePair>& sel, const ImVec4& brush_col)
  {
    if (brush_seen.empty())
    {
      ui.persistence_bins.clear();
      ui.persistence_bin_colors.clear();
    }

    ImU32 ucol = ImGui::ColorConvertFloat4ToU32(brush_col);
    for (auto &p : sel)
    {
      int bs = std::clamp<int>(p.birth, 0, AppState::TF2D_BINS - 1);
      int ds = std::clamp<int>(p.death, 0, AppState::TF2D_BINS - 1);
      if (bs > ds) std::swap(bs, ds);
      if (bs == ds)
      {
        bs = std::max(bs - 1.0, 0.0);
        ds = std::min(ds + 1.0, AppState::TF2D_BINS - 1.0);
      }

      for (int s = bs; s <= ds; ++s)
      {
        for (int g : grads_by_scalar[s])
        {
          int fg = (AppState::TF2D_BINS - 1) - g;
          uint32_t key = (uint32_t(s) << 16)|uint32_t(fg);
          if (brush_seen.insert(key).second)
          {
            ui.persistence_bins.emplace_back(s,fg);
            ui.persistence_bin_colors.push_back(ucol);
          }
        }
      }
    }
  });

  ui.set_on_evaluation([&](float J_arc, float J_box, float prec, float rec)
  {
    ui.last_J_arc         = J_arc;
    ui.last_J_box         = J_box;
    ui.last_precision     = prec;
    ui.last_recall        = rec;
    ui.last_metrics_valid = true;
  });

  ui.set_on_range_applied([this](const std::vector<PersistencePair>& sel)
  {
    if (sel.empty()) return;
    // we only ever get one pair here on a click
    const auto &p = sel[0];

    // turn it into a “volume highlight” request
    std::vector<std::pair<PersistencePair,float>> hits {{ p, 1.0f }};
    int ramp = ui.get_selected_ramp();

    // reuse your existing volume‐highlight code
    this->volume_highlight_persistence_pairs(hits, ramp);
  });

  ui.set_on_multi_selected([this](const std::vector<PersistencePair>& sel)
  {
    if (sel.empty()) return;
    std::vector<std::pair<PersistencePair,float>> hits;
    hits.reserve(sel.size());
    for (auto &p : sel) hits.emplace_back(p, 1.0f);
    int ramp = ui.get_selected_ramp();
    this->volume_highlight_persistence_pairs(hits, ramp);
  });

  export_persistence_pairs_to_csv(persistence_pairs, gradient_persistence_pairs, "scalar_pairs.csv", "gradient_pairs.csv");
    // scalar volume
    std::ofstream outS("volume_data/scalar_volume.bin", std::ios::binary);
    outS.write(reinterpret_cast<const char*>(scalar_volume->data.data()), scalar_volume->data.size() * sizeof(scalar_volume->data[0]));

    // gradient volume
    std::ofstream outG("volume_data/gradient_volume.bin", std::ios::binary);
    outG.write(reinterpret_cast<const char*>(gradient_volume.data.data()), gradient_volume.data.size() * sizeof(gradient_volume.data[0]));

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
  if (pending_reproject_idx >= 0)
  {
  // rebuild tf_data right now for that one feature
  PersistencePair p = (ui.get_pd_mode()==1 ? gradient_persistence_pairs[pending_reproject_idx] : persistence_pairs[pending_reproject_idx]);
  std::vector<std::pair<PersistencePair,float>> single{{p,1.0f}};
  volume_highlight_persistence_pairs(single, ui.get_selected_ramp());
  pending_reproject_idx = -1;
}
  
  auto &buf = storage.get_buffer_by_name("transfer_function");
  buf.update_data(tf_data);
  vmc.logical_device.get().waitIdle();

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

  vk::CommandBuffer& compute_cb = vcc.begin(vcc.compute_cbs[0]);
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

void WorkContext::set_persistence_pairs(std::vector<PersistencePair> pairs, const Volume& volume)
{
  persistence_pairs = std::move(pairs);

  // compute the global max persistence, later used in isolate/volumeHighlight
  global_max_persistence = 1;
  for (auto &p : persistence_pairs)
  {
    uint32_t pers = (p.death > p.birth ? p.death - p.birth : 0);
    global_max_persistence = std::max(global_max_persistence, pers);
  }

  transfer_function.update(persistence_pairs, volume, tf_data);
}

void WorkContext::set_gradient_persistence_pairs(const std::vector<PersistencePair>& pairs)
{
  gradient_persistence_pairs = pairs;
  ui.set_gradient_persistence_pairs(&gradient_persistence_pairs);
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

void WorkContext::volume_highlight_persistence_pairs(const std::vector<std::pair<PersistencePair, float>>& pairs, int ramp_index)
{
  tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));

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
        float hue = (1.0f - tColor) * (240.0f/360.0f);
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
        float hue = (1.0f - tColor) * (240.0f/360.0f);
        ImGui::ColorConvertHSVtoRGB(hue, 1.0, 1.0, rgb.x, rgb.y, rgb.z);
      }
    }

    float alpha = brush_op;
    if (ramp_index == UI::RAMP_CUSTOM)
    {
      alpha *= ui.get_custom_falloff();
    }

    int B = AppState::TF2D_BINS;
    uint32_t bi = std::clamp(p.birth, 0u, B-1u);
    uint32_t di = std::clamp(p.death, 0u, B-1u);
    if (bi > di) std::swap(bi, di);

    // primary
    uint32_t s0 = bi, s1 = di;
    if (ui.tf2d_use_primary)
    {
      s0 = ui.tf2d_primary_min;
      s1 = ui.tf2d_primary_max;
    }

    // gradient‐axis bounds
    uint32_t g0 = 0, g1 = B-1;
    if (ui.tf2d_use_secondary)
    {
      // secondary slider always clamps the gradient axis in scalar mode
      g0 = ui.tf2d_secondary_min;
      g1 = ui.tf2d_secondary_max;
    }

    for (uint32_t g = g0; g <= g1; ++g)
    {
      uint32_t fg   = (B - 1) - g;
      uint32_t base = fg * B;
      for (uint32_t s = s0; s <= s1; ++s)
      {
        tf_data[base + s] = glm::vec4(rgb, alpha);
      }
    }
  }

  // reapply custom_colors exactly as before
  for (auto& assign : custom_colors)
  {
    const auto& p = assign.first;
    const auto& col = assign.second;
    uint32_t bi = std::clamp(p.birth, 0u, (uint32_t)AppState::TF2D_BINS - 1);
    uint32_t di = std::clamp(p.death, 0u, (uint32_t)AppState::TF2D_BINS - 1);
    if (bi > di) std::swap(bi, di);

    uint32_t g0 = 0, g1 = AppState::TF2D_BINS-1;
    if (ui.tf2d_use_secondary)
    {
      g0 = ui.tf2d_secondary_min;
      g1 = ui.tf2d_secondary_max;
    }

    for (uint32_t g = g0; g <= g1; ++g)
    {
      uint32_t base = g * AppState::TF2D_BINS;
      for (uint32_t s = bi; s <= di; ++s)
      {
          tf_data[base + s] = col;
      }
    }
  }
}

std::pair<uint32_t, uint32_t> WorkContext::clamp_and_sort_range(const PersistencePair& p)
{
  constexpr uint32_t maxBin = AppState::TF2D_BINS - 1u;
  uint32_t low = (p.birth < AppState::TF2D_BINS ? p.birth : maxBin);
  uint32_t high = (p.death < AppState::TF2D_BINS ? p.death : maxBin);
  if (low > high) std::swap(low, high);
  return { low, high };
}

void WorkContext::highlight_diff(const PersistencePair &base, const PersistencePair &mask)
{
  tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));

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
    glm::vec4 col(c.x, c.y, c.z, c.w);
    for (uint32_t g = 0; g < AppState::TF2D_BINS; ++g)
    {
      uint32_t base_idx = g * AppState::TF2D_BINS;
      for (uint32_t s = b0; s <= d0; ++s)
      {
        tf_data[base_idx + s] = col;
      }
    }
  }

  // mask out [b1..d1] (transparent)
  for (uint32_t g = 0; g < AppState::TF2D_BINS; ++g)
  {
    uint32_t baseIdx = g * AppState::TF2D_BINS;
    for (uint32_t s = b1; s <= d1; ++s)
    {
      tf_data[baseIdx + s] = glm::vec4(0.0f);
    }
  }
}

void WorkContext::highlight_intersection(const PersistencePair& a, const PersistencePair& b)
{
  tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));

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
    glm::vec4 col(c.x, c.y, c.z, c.w);
    for (uint32_t g = 0; g < AppState::TF2D_BINS; ++g)
    {
      uint32_t base_idx = g * AppState::TF2D_BINS;
      for (uint32_t s = start; s <= end; ++s)
      {
        tf_data[base_idx + s] = col;
      }
    }
  }

  // paint A-only before/after
  if (ui.intersect_enabled_Aonly)
  {
    ImVec4 cA = ui.intersect_color_Aonly;
    glm::vec4 colA{cA.x, cA.y, cA.z, cA.w};
    for (uint32_t g = 0; g < AppState::TF2D_BINS; ++g)
    {
      uint32_t base_idx = g * AppState::TF2D_BINS;
      for (uint32_t s = a0; s < start; ++s)
      {
        tf_data[base_idx + s] = colA;
      }
      for (uint32_t s = end + 1; s <= a1; ++s)
      {
        tf_data[base_idx + s] = colA;
      }
    }
  }

  // paint B-only before/after
  if (ui.intersect_enabled_Bonly)
  {
    ImVec4 cB = ui.intersect_color_Bonly;
    glm::vec4 colB{cB.x, cB.y, cB.z, cB.w};
    for (uint32_t g = 0; g < AppState::TF2D_BINS; ++g)
    {
      uint32_t base_idx = g * AppState::TF2D_BINS;
      for (uint32_t s = b0; s < start; ++s)
      {
        tf_data[base_idx + s] = colB;
      }
      for (uint32_t s = end + 1; s <= b1; ++s)
      {
        tf_data[base_idx + s] = colB;
      }
    }
  }
}

void WorkContext::highlight_union(const PersistencePair& a, const PersistencePair& b)
{
    tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));

    auto [a0,a1] = clamp_and_sort_range(a);
    auto [b0,b1] = clamp_and_sort_range(b);
    uint32_t start = std::max(a0, b0);
    uint32_t end = std::min(a1, b1);
    if (ui.union_enabled_Aonly)
    {
        ImVec4 cA = ui.union_color_Aonly;
        glm::vec4 colA{cA.x, cA.y, cA.z, cA.w};
        for (int g = 0; g < AppState::TF2D_BINS; ++g)
        {
            uint32_t base = g * AppState::TF2D_BINS;
            for (uint32_t s = a0; s < start; ++s)
                tf_data[base + s] = colA;
            for (uint32_t s = end + 1; s <= a1; ++s)
                tf_data[base + s] = colA;
        }
    }
    if (ui.union_enabled_Bonly)
    {
        ImVec4 cB = ui.union_color_Bonly;
        glm::vec4 colB{cB.x, cB.y, cB.z, cB.w};
        for (int g = 0; g < AppState::TF2D_BINS; ++g)
        {
            uint32_t base = g * AppState::TF2D_BINS;
            for (uint32_t s = b0; s < start; ++s)
                tf_data[base + s] = colB;
            for (uint32_t s = end + 1; s <= b1; ++s)
                tf_data[base + s] = colB;
        }
    }

    if (ui.union_enabled_common)
    {
        uint32_t u0 = std::min(a0, b0);
        uint32_t u1 = std::max(a1, b1);
        ImVec4 cU = ui.union_color_common;
        glm::vec4 colU{cU.x, cU.y, cU.z, cU.w};
        for (int g = 0; g < AppState::TF2D_BINS; ++g)
        {
            uint32_t base = g * AppState::TF2D_BINS;
            for (uint32_t s = u0; s <= u1; ++s)
                tf_data[base + s] = colU;
        }
    }
}

void WorkContext::highlight_onlyA(const PersistencePair& a, const PersistencePair&, const ImVec4& c)
{
  tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));
  auto [a0,a1] = clamp_and_sort_range(a);
  glm::vec4 col{c.x, c.y, c.z, c.w};
  for (int g = 0; g < AppState::TF2D_BINS; ++g)
      for (uint32_t s = a0; s <= a1; ++s)
          tf_data[g * AppState::TF2D_BINS + s] = col;
}

void WorkContext::highlight_onlyB(const PersistencePair&, const PersistencePair& b, const ImVec4& c)
{
  tf_data.assign(AppState::TF2D_BINS * AppState::TF2D_BINS, glm::vec4(0.0f));
  auto [b0,b1] = clamp_and_sort_range(b);
  glm::vec4 col{c.x, c.y, c.z, c.w};
  for (int g = 0; g < AppState::TF2D_BINS; ++g)
      for (uint32_t s = b0; s <= b1; ++s)
          tf_data[g * AppState::TF2D_BINS + s] = col;
}

void WorkContext::export_persistence_pairs_to_csv(const std::vector<PersistencePair>& scalar_pairs, const std::vector<PersistencePair>& gradient_pairs, const std::string& scalar_filename, const std::string& gradient_filename) const
{
    // ensure output directory exists
    if (mkdir("volume_data", 0755) != 0 && errno != EEXIST)
    {
        std::cerr << "Error: could not create directory 'volume_data'";
        return;
    }

    std::string scalar_path   = std::string("volume_data/") + scalar_filename;
    std::string gradient_path = std::string("volume_data/") + gradient_filename;

    Timer<float> local_timer;
    using ms = std::milli;

    // scalar-mode pairs to CSV
    {
        std::ofstream out_scalar(scalar_path);
        if (!out_scalar)
        {
            std::cerr << "Error: could not open '" << scalar_filename << "' for writing";
            return;
        }

       auto t0 = local_timer.restart<ms>();
       out_scalar << "birth,death\n";
       for (const auto& p : scalar_pairs)
        out_scalar << p.birth << "," << p.death << "\n"; 
        auto t1 = local_timer.restart<ms>();
        std::cout << CLR_GREEN << "[TIMING] Scalar export: " << t1 << " ms" << std::endl << CLR_RESET;
    }

    // gradient-mode pairs to CSV
    {
        std::ofstream out_grad(gradient_path);
        if (!out_grad)
        {
            std::cerr << "Error: could not open '" << gradient_filename << "' for writing";
            return;
        }

        auto t2 = local_timer.restart<ms>();
        out_grad << "birth,death";
        for (const auto& p : gradient_pairs)
        {
            out_grad << p.birth << "," << p.death << "";
        }
        auto t3 = local_timer.restart<ms>();
        std::cout << CLR_GREEN << "[TIMING] Gradient export: " << t3 << " ms" << std::endl << CLR_RESET;
    }

    std::cout << "Exported persistence pairs to:" << "  - " << scalar_filename  << ""  << "  - " << gradient_filename << "" << std::endl;
}

void WorkContext::apply_custom_color_to_volume(const std::vector<PersistencePair>& pairs, const ImVec4& color)
{
  glm::vec4 chosen_color(color.x, color.y, color.z, color.w);

  // record every new assignment
  for (auto &p : pairs)
  {
      custom_colors.emplace_back(p, chosen_color);
  }

  // replay all custom assignments
  for (const auto &assign : custom_colors)
  {
    const PersistencePair &p = assign.first;
    const glm::vec4 &col = assign.second;

    uint32_t b = std::clamp(p.birth, 0u, AppState::TF2D_BINS - 1);
    uint32_t d = std::clamp(p.death, 0u, AppState::TF2D_BINS - 1);
    if (b > d) std::swap(b, d);

    for (uint32_t g = 0; g < AppState::TF2D_BINS; ++g)
    {
      uint32_t base_idx = g * AppState::TF2D_BINS;
      for (uint32_t s = b; s <= d; ++s)
      {
          tf_data[base_idx + s] = col;
      }
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

  volume_highlight_persistence_pairs(all_hits, ramp);
}

void WorkContext::reproject_and_compare()
{
  // build two independent bin‐masks:
  // A_mask = manual TF2D selection from the UI (last_tf2d_bins)
  // P_mask = persistence reprojection mask stored in ui.persistence_bins
  std::vector<bool> A_mask(AppState::TF2D_BINS * AppState::TF2D_BINS, false), P_mask(AppState::TF2D_BINS * AppState::TF2D_BINS, false);
  for (auto &b : last_tf2d_bins)
      A_mask[b.second * AppState::TF2D_BINS + b.first] = true;
  for (auto &b : ui.persistence_bins) // the reprojed persistence bins
      P_mask[b.second * AppState::TF2D_BINS + b.first] = true;

  // lift those to voxel‐level masks
  const auto& vol  = *scalar_volume;
  const auto& grad = gradient_volume;
  size_t Nvox = vol.data.size();
  std::vector<bool> voxA(Nvox,false), voxP(Nvox,false);

  for (size_t i = 0; i < Nvox; ++i)
  {
      int s  = int(vol.data[i]);
      int g  = int(grad.data[i]);
      int fg = (AppState::TF2D_BINS - 1) - g;
      if (s >= 0 && s < AppState::TF2D_BINS && fg >= 0 && fg < AppState::TF2D_BINS)
      {
          int idx = fg * AppState::TF2D_BINS + s;
          voxA[i] = A_mask[idx];
          voxP[i] = P_mask[idx];
      }
  }

  // compute voxel‐level J_arc (Jaccard), precision, recall
  size_t countA = 0, countP = 0, intersect = 0, uni = 0;
  for (size_t i = 0; i < Nvox; ++i)
  {
    bool a = voxA[i], p = voxP[i];
    if (a && p) ++intersect;
    if (a || p) ++uni;
    if (a) ++countA;
    if (p) ++countP;
  }
  float J_arc = float(intersect) / float(uni + 1e-6f);
  float precision = float(intersect) / float(countP + 1e-6f);
  float recall = float(intersect) / float(countA + 1e-6f);

  // compute the tight axis‐aligned bounding‐box of P_mask in bin‐space
  int smin = AppState::TF2D_BINS;
  int smax = -1;
  int gmin =  AppState::TF2D_BINS;
  int gmax = -1;

  for (int g = 0; g < AppState::TF2D_BINS; ++g)
  {
    for (int s = 0; s < AppState::TF2D_BINS; ++s)
    {
      if (P_mask[g * AppState::TF2D_BINS + s])
      {
        smin = std::min(smin, s);
        smax = std::max(smax, s);
        gmin = std::min(gmin, g);
        gmax = std::max(gmax, g);
      }
    }
  }
  if (smax < smin || gmax < gmin)
  {
    // no bins -> collapse to a single cell
    smin = smax = gmin = gmax = 0;
  }

  // measure J_box over voxels
  size_t countBox = 0, box_and_P = 0;
  for (size_t i = 0; i < Nvox; ++i)
  {
    int s  = int(vol.data[i]);
    int g  = int(grad.data[i]);
    int fg = (AppState::TF2D_BINS - 1) - g;
    if (s >= 0 && s < AppState::TF2D_BINS && fg >= 0 && fg < AppState::TF2D_BINS)
    {
      bool inBox = (s >= smin && s <= smax && fg >= gmin && fg <= gmax);
      bool p = voxP[i];
      if (inBox) ++countBox;
      if (inBox && p) ++box_and_P;
    }
  }
  float J_box = float(box_and_P) / float(countBox + countP - box_and_P + 1e-6f);

  std::cout << "[Reprojection] "
            << "J_arc="      << J_arc
            << "  J_box="    << J_box
            << "  Precision="<< precision
            << "  Recall="   << recall
            << "  |A|="      << countA
            << "  |P|="      << countP
            << "\n";

  if (ui.on_evaluation)
  {
      ui.on_evaluation(J_arc, J_box, precision, recall);
  }
}
}//namespace ve