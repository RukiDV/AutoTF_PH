#include "window.hpp"

#include "vk/ve_log.hpp"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

Window::Window(const uint32_t width, const uint32_t height)
{
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
  window = SDL_CreateWindow("AutoTF_PH", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI);
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
  std::vector<const char*> extensions;
  uint32_t extension_count;
  VE_ASSERT(SDL_Vulkan_GetInstanceExtensions(window, &extension_count, nullptr), "Failed to load extension count for window!");
  extensions.resize(extension_count);
  VE_ASSERT(SDL_Vulkan_GetInstanceExtensions(window, &extension_count, extensions.data()), "Failed to load required extensions for window!");
  return extensions;
}

vk::SurfaceKHR Window::create_surface(const vk::Instance& instance)
{
  vk::SurfaceKHR surface;
  VE_ASSERT(SDL_Vulkan_CreateSurface(window, instance, reinterpret_cast<VkSurfaceKHR*>(&surface)), "Failed to create surface!");
  return surface;
}
