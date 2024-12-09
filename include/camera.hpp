#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

class Camera
{
public:
    Camera();
    void update_data();
    void update();
    void translate(glm::vec3 v);
    void on_mouse_move(glm::vec2 move);
    void move_front(float amount);
    void move_right(float amount);
    void move_up(float amount);
    void rotate(float amount);
    void update_screen_size(float aspect_ratio);
    const glm::vec3& get_position() const;
    float get_near() const;
    float get_far() const;

    struct Data
    {
        alignas(16) glm::vec3 pos;
        alignas(16) glm::vec3 u;
        alignas(16) glm::vec3 v;
        alignas(16) glm::vec3 w;
    } data;

private:
    glm::vec3 position;
    glm::quat orientation;
    glm::vec3 u, v, w;
    float yaw, pitch;
};
