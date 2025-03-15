#include "gpu_renderer.hpp"

#include "event_handler.hpp"
#include "work_context.hpp"
#include "util/timer.hpp"
#include "SDL3/SDL_mouse.h"

struct GPUContext
{
    GPUContext(AppState& app_state, const Volume& volume) : vcc(vmc), wc(vmc, vcc)
    {
        vmc.construct(app_state.get_window_extent().width, app_state.get_window_extent().height);
        vcc.construct();
        wc.construct(app_state, volume);
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

void dispatch_pressed_keys(GPUContext& gpu_context, EventHandler& eh, AppState& app_state)
{
    float move_amount = app_state.time_diff * app_state.move_speed;
    if (eh.is_key_pressed(Key::W)) app_state.cam.move_front(move_amount);
    if (eh.is_key_pressed(Key::A)) app_state.cam.move_right(-move_amount);
    if (eh.is_key_pressed(Key::S)) app_state.cam.move_front(-move_amount);
    if (eh.is_key_pressed(Key::D)) app_state.cam.move_right(move_amount);
    if (eh.is_key_pressed(Key::Q)) app_state.cam.move_up(-move_amount);
    if (eh.is_key_pressed(Key::E)) app_state.cam.move_up(move_amount);
    float panning_speed = eh.is_key_pressed(Key::Shift) ? 50.0f : 200.0f;
    if (eh.is_key_pressed(Key::Left)) app_state.cam.on_mouse_move(glm::vec2(-panning_speed * app_state.time_diff, 0.0f));
    if (eh.is_key_pressed(Key::Right)) app_state.cam.on_mouse_move(glm::vec2(panning_speed * app_state.time_diff, 0.0f));
    if (eh.is_key_pressed(Key::Up)) app_state.cam.on_mouse_move(glm::vec2(0.0f, -panning_speed * app_state.time_diff));
    if (eh.is_key_pressed(Key::Down)) app_state.cam.on_mouse_move(glm::vec2(0.0f, panning_speed * app_state.time_diff));

    // reset state of keys that are used to execute a one time action
    if (eh.is_key_released(Key::Plus))
    {
        app_state.move_speed *= 2.0f;
        eh.set_released_key(Key::Plus, false);
    }
    if (eh.is_key_released(Key::Minus))
    {
        app_state.move_speed /= 2.0f;
        eh.set_released_key(Key::Minus, false);
    }
    if (eh.is_key_released(Key::G))
    {
        app_state.show_ui = !app_state.show_ui;
        eh.set_released_key(Key::G, false);
    }
    if (eh.is_key_released(Key::F1))
    {
        app_state.save_screenshot = true;
        eh.set_released_key(Key::F1, false);
    }
    if (eh.is_key_pressed(Key::MouseLeft))
    {
        if (!SDL_GetWindowRelativeMouseMode(gpu_context.vmc.window->get()))
        {
            SDL_SetWindowRelativeMouseMode(gpu_context.vmc.window->get(), true);
        }
        app_state.cam.on_mouse_move(glm::vec2(eh.mouse_motion.x * 1.5f, eh.mouse_motion.y * 1.5f));
        eh.mouse_motion = glm::vec2(0.0f);
    }
    if (eh.is_key_released(Key::MouseLeft))
    {
        SDL_SetWindowRelativeMouseMode(gpu_context.vmc.window->get(), false);
        SDL_WarpMouseInWindow(gpu_context.vmc.window->get(), app_state.get_window_extent().width / 2.0f, app_state.get_window_extent().height / 2.0f);
        eh.set_released_key(Key::MouseLeft, false);
    }
}

int gpu_render(const Volume& volume)
{
    AppState app_state;
    EventHandler eh;
    GPUContext gpu_context(app_state, volume);
    bool quit = false;
    Timer rendering_timer;
    SDL_Event e;
    while (!quit)
    {
        dispatch_pressed_keys(gpu_context, eh, app_state);
        app_state.cam.update();

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
            quit |= e.window.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED;
            eh.dispatch_event(e);
        }
        app_state.time_diff = rendering_timer.restart();
    }

    return 0;
}
