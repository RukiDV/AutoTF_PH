#include "event_handler.hpp"

#include "backends/imgui_impl_sdl2.h"

EventHandler::EventHandler() : pressed_keys(get_idx(Key::Size), false), released_keys(get_idx(Key::Size), false)
{}

void EventHandler::dispatch_event(SDL_Event e)
{
  ImGui_ImplSDL2_ProcessEvent(&e);
  if (ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard)
  {
    return;
  }
  switch (e.type)
    {
      case SDL_MOUSEMOTION:
          mouse_motion.x = e.motion.xrel;
          mouse_motion.y = e.motion.yrel;
          break;
    }
  switch (e.button.button)
  {
    case SDL_BUTTON_LEFT:
      apply_key_event(Key::MouseLeft, e.type);
      break;
    case SDL_BUTTON_MIDDLE:
      apply_key_event(Key::MouseMiddle, e.type);
      break;
    case SDL_BUTTON_RIGHT:
      apply_key_event(Key::MouseRight, e.type);
      break;
  }
  switch (e.key.keysym.sym)
  {
    case SDLK_a:
      apply_key_event(Key::A, e.type);
      break;
    case SDLK_b:
      apply_key_event(Key::B, e.type);
      break;
    case SDLK_c:
      apply_key_event(Key::C, e.type);
      break;
    case SDLK_d:
      apply_key_event(Key::D, e.type);
      break;
    case SDLK_e:
      apply_key_event(Key::E, e.type);
      break;
    case SDLK_f:
      apply_key_event(Key::F, e.type);
      break;
    case SDLK_g:
      apply_key_event(Key::G, e.type);
      break;
    case SDLK_h:
      apply_key_event(Key::H, e.type);
      break;
    case SDLK_i:
      apply_key_event(Key::I, e.type);
      break;
    case SDLK_j:
      apply_key_event(Key::J, e.type);
      break;
    case SDLK_k:
      apply_key_event(Key::K, e.type);
      break;
    case SDLK_l:
      apply_key_event(Key::L, e.type);
      break;
    case SDLK_m:
      apply_key_event(Key::M, e.type);
      break;
    case SDLK_n:
      apply_key_event(Key::N, e.type);
      break;
    case SDLK_o:
      apply_key_event(Key::O, e.type);
      break;
    case SDLK_p:
      apply_key_event(Key::P, e.type);
      break;
    case SDLK_q:
      apply_key_event(Key::Q, e.type);
      break;
    case SDLK_r:
      apply_key_event(Key::R, e.type);
      break;
    case SDLK_s:
      apply_key_event(Key::S, e.type);
      break;
    case SDLK_t:
      apply_key_event(Key::T, e.type);
      break;
    case SDLK_u:
      apply_key_event(Key::U, e.type);
      break;
    case SDLK_v:
      apply_key_event(Key::V, e.type);
      break;
    case SDLK_w:
      apply_key_event(Key::W, e.type);
      break;
    case SDLK_x:
      apply_key_event(Key::X, e.type);
      break;
    case SDLK_y:
      apply_key_event(Key::Y, e.type);
      break;
    case SDLK_z:
      apply_key_event(Key::Z, e.type);
      break;
    case SDLK_KP_PLUS:
      apply_key_event(Key::Plus, e.type);
      break;
    case SDLK_KP_MINUS:
      apply_key_event(Key::Minus, e.type);
      break;
    case SDLK_LEFT:
      apply_key_event(Key::Left, e.type);
      break;
    case SDLK_RIGHT:
      apply_key_event(Key::Right, e.type);
      break;
    case SDLK_UP:
      apply_key_event(Key::Up, e.type);
      break;
    case SDLK_DOWN:
      apply_key_event(Key::Down, e.type);
      break;
    case SDLK_LSHIFT:
      apply_key_event(Key::Shift, e.type);
      break;
    case SDLK_RSHIFT:
      apply_key_event(Key::Shift, e.type);
      break;
    case SDLK_F1:
      apply_key_event(Key::F1, e.type);
      break;
    case SDLK_0:
      apply_key_event(Key::Zero, e.type);
      break;
    case SDLK_1:
      apply_key_event(Key::One, e.type);
      break;
    case SDLK_2:
      apply_key_event(Key::Two, e.type);
      break;
    case SDLK_3:
      apply_key_event(Key::Three, e.type);
      break;
    case SDLK_4:
      apply_key_event(Key::Four, e.type);
      break;
    case SDLK_5:
      apply_key_event(Key::Five, e.type);
      break;
    case SDLK_6:
      apply_key_event(Key::Six, e.type);
      break;
    case SDLK_7:
      apply_key_event(Key::Seven, e.type);
      break;
    case SDLK_8:
      apply_key_event(Key::Eight, e.type);
      break;
    case SDLK_9:
      apply_key_event(Key::Nine, e.type);
      break;
    case SDLK_RETURN:
      apply_key_event(Key::Return, e.type);
      break;
  }
}

bool EventHandler::is_key_pressed(Key key) const
{
  return pressed_keys[get_idx(key)];
}

bool EventHandler::is_key_released(Key key) const
{
  return released_keys[get_idx(key)];
}

void EventHandler::set_pressed_key(Key key, bool value)
{
  pressed_keys[get_idx(key)] = value;
}

void EventHandler::set_released_key(Key key, bool value)
{
  released_keys[get_idx(key)] = value;
}

void EventHandler::apply_key_event(Key k, uint32_t et)
{
  if (et == SDL_KEYDOWN || et == SDL_MOUSEBUTTONDOWN)
  {
    pressed_keys[get_idx(k)] = true;
    released_keys[get_idx(k)] = false;
  }
  else if (et == SDL_KEYUP || et == SDL_MOUSEBUTTONUP)
  {
    pressed_keys[get_idx(k)] = false;
    released_keys[get_idx(k)] = true;
  }
}

uint32_t EventHandler::get_idx(Key key)
{
  return static_cast<uint32_t>(key);
}
