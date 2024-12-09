#pragma once
#include <cstdint>
#include "camera.hpp"
#include "vk/common.hpp"

struct AppState {
public:
  Camera cam;
  uint32_t current_frame = 0;
  uint32_t total_frames = 0;
  bool vsync = true;
  bool show_ui = true;
  float time_diff = 0.000001f;
  float move_speed = 10.0f;
  bool save_screenshot = false;

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

