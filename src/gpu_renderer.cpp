#include "gpu_renderer.hpp"

#include "SDL_events.h"
#include "event_handler.hpp"
#include "work_context.hpp"
#include "util/timer.hpp"

struct GPUContext
{
  GPUContext(AppState& app_state, const Scene& scene) : vmc(), vcc(vmc), wc(vmc, vcc)
  {
    vmc.construct(app_state.get_window_extent().width, app_state.get_window_extent().height);
    vcc.construct();
    wc.construct(app_state, scene);
  }

  ~GPUContext()
  {
    wc.destruct();
    vcc.destruct();
    vmc.destruct();
  }
  ve::VulkanMainContext vmc;
  ve::VulkanCommandContext vcc;
  ve::WorkContext wc;
};

void dispatch_pressed_keys(EventHandler& event_handler, AppState& app_state)
{
  if (event_handler.is_key_released(Key::G))
  {
    app_state.show_ui = !app_state.show_ui;
    event_handler.set_released_key(Key::G, false);
  }
}

int gpu_render(const Scene& scene)
{
  AppState app_state;
  app_state.set_render_extent(vk::Extent2D(scene.resolution.x, scene.resolution.y));
  GPUContext gpu_context(app_state, scene);
  EventHandler event_handler;

  bool quit = false;
  Timer rendering_timer;
  SDL_Event e;
  while (!quit)
  {
    dispatch_pressed_keys(event_handler, app_state);
    try
    {
      gpu_context.wc.draw_frame(app_state);
    }
    catch (const vk::OutOfDateKHRError e)
    {
      app_state.set_window_extent(gpu_context.wc.recreate_swapchain(app_state.vsync));
    }
    while (SDL_PollEvent(&e))
    {
      quit |= e.window.event == SDL_WINDOWEVENT_CLOSE;
      event_handler.dispatch_event(e);
    }
    std::cout << "frametime: " << rendering_timer.restart<std::milli>() << std::endl;
  }

  return 0;
}
