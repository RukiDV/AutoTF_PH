#pragma once

#include "vk/common.hpp"
#include "SDL3/SDL_video.h"

#include <vector>

class Window
{
public:
  Window(const uint32_t width, const uint32_t height);
  void destruct();
  SDL_Window* get() const;
  std::vector<const char*> get_required_extensions() const;
  vk::SurfaceKHR create_surface(const vk::Instance& instance);

private:
  SDL_Window* window;
};
