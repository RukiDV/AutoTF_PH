#include "window.hpp"

#include "vk/ve_log.hpp"
#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

Window::Window(const uint32_t width, const uint32_t height)
{
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
  window = SDL_CreateWindow("AutoTF_PH", width, height, SDL_WINDOW_VULKAN);
  std::cout << height << "," << width << std::endl;
}

void Window::destruct()
{
  SDL_DestroyWindow(window);
  SDL_Quit();
}

SDL_Window* Window::get() const
{
  return window;
}

std::vector<const char*> Window::get_required_extensions() const
{
  uint32_t extension_count;
  const char* const* extensions_sdl = SDL_Vulkan_GetInstanceExtensions(&extension_count);
  std::vector<const char*> extensions;
  for (int i = 0; i < extension_count; i++)
  {
    extensions.push_back(extensions_sdl[i]);
  }
  return extensions;
}

vk::SurfaceKHR Window::create_surface(const vk::Instance& instance)
{
  vk::SurfaceKHR surface;
  VE_ASSERT(SDL_Vulkan_CreateSurface(window, instance, nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface)), "Failed to create surface!");
  return surface;
}
