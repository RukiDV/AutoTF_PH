#pragma once
#include <cstdint>
#include "vk/common.hpp"
#include "camera.hpp"
#include "vk/device_timer.hpp"
#include "volume.hpp"
#include "persistence.hpp"

struct AppState {
public:
  Camera cam;
  uint32_t total_frames = 0;
  std::array<float, ve::DeviceTimer::TIMER_COUNT> device_timings;
  bool vsync = true;
  bool show_ui = true;
  float time_diff = 0.000001f;
  float move_speed = 10.0f;
  bool save_screenshot = false;

 // controlling levels in the merge tree
  int target_level = 0;
  bool apply_target_level = false;

  // threshold-based cuts in persistent homology
  int persistence_threshold = 0;
  bool apply_persistence_threshold = false;

  // applying a purely histogram-based transfer function
  bool apply_histogram_tf = false;

  // hybrid histogram + PH approach.
  bool apply_histogram_ph_tf = false;
  int ph_threshold = 10;

  FiltrationMode filtration_mode = FiltrationMode::LowerStar;
  bool apply_filtration_mode = false;

  bool apply_highlight_update = false;
  PersistencePair selected_pair; 
  
  // mode for iso-surface = 0 or volume highlight = 1 
  int display_mode = 1;
  float max_gradient = 0.0f;
  float density_threshold = 0.0f;

  static constexpr uint32_t TF2D_BINS = 256;

  vk::Extent2D get_render_extent() const { return render_extent; }
  vk::Extent2D get_window_extent() const { return window_extent; }
  float get_aspect_ratio() const { return aspect_ratio; }

  void set_render_extent(vk::Extent2D extent)
  {
    render_extent = extent;
    aspect_ratio = float(render_extent.width) / float(render_extent.height);
    window_extent = vk::Extent2D(aspect_ratio * 1000, 1000);
  }

  void set_window_extent(vk::Extent2D extent)
  {
    window_extent = extent;
  }

private:
  vk::Extent2D render_extent = vk::Extent2D(1920, 1080);
  float aspect_ratio = float(render_extent.width) / float(render_extent.height);
  vk::Extent2D window_extent = vk::Extent2D(aspect_ratio * 1000, 1000);
};

