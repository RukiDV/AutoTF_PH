#include "camera.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

inline constexpr glm::vec3 back(0.0f, 0.0f, 1.0f);
inline constexpr glm::vec3 right(1.0f, 0.0f, 0.0f);
inline constexpr glm::vec3 up(0.0f, 1.0f, 0.0f);

Camera::Camera() : yaw(0.0f), pitch(0.0f), position(0.0f, 0.0f, 5.0f)
{
    orientation = glm::quatLookAt(-back, up);
}

void Camera::update_data()
{
    data.pos = position;
    data.u = u;
    data.v = v;
    data.w = w;
}

void Camera::update()
{
    // rotate initial coordinate system to camera orientation
    glm::quat q_back = glm::normalize(orientation * glm::quat(0.0f, back) * glm::conjugate(orientation));
    glm::quat q_right = glm::normalize(orientation * glm::quat(0.0f, right) * glm::conjugate(orientation));
    glm::quat q_up = glm::normalize(orientation * glm::quat(0.0f, up) * glm::conjugate(orientation));
    w = glm::normalize(glm::vec3(q_back.x, q_back.y, q_back.z));
    u = glm::normalize(glm::vec3(q_right.x, q_right.y, q_right.z));
    v = glm::normalize(glm::vec3(q_up.x, q_up.y, q_up.z));

    pitch = glm::clamp(pitch, -89.0f, 89.0f);
    glm::quat q_pitch = glm::angleAxis(glm::radians(pitch), right);
    glm::quat q_yaw = glm::angleAxis(glm::radians(yaw), up);
    orientation = glm::normalize(q_yaw * q_pitch);
}

void Camera::translate(glm::vec3 amount)
{
    position += amount;
}

void Camera::on_mouse_move(glm::vec2 move)
{
    yaw -= move.x;
    pitch -= move.y;
}

void Camera::move_front(float amount)
{
    translate(-amount * w);
}

void Camera::move_right(float amount)
{
    translate(amount * u);
}

void Camera::move_up(float amount)
{
    translate(up * amount);
}

const glm::vec3& Camera::get_position() const
{
    return position;
}
