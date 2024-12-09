cmake_minimum_required(VERSION 3.19)
project(AutoTF_PH)
set(CMAKE_CXX_STANDARD 20)

set(SDL2_DIR "${PROJECT_SOURCE_DIR}/dependencies/SDL2")

set(SOURCE_FILES
  src/gpu_renderer.cpp
  src/volume.cpp
  src/util/random_generator.cpp
  src/vk/command_pool.cpp
  src/vk/descriptor_set_handler.cpp
  src/vk/extensions_handler.cpp
  src/vk/image.cpp
  src/vk/instance.cpp
  src/vk/logical_device.cpp
  src/vk/physical_device.cpp
  src/vk/pipeline.cpp
  src/vk/renderer.cpp
  src/vk/render_pass.cpp
  src/vk/shader.cpp
  src/vk/storage.cpp
  src/vk/swapchain.cpp
  src/vk/synchronization.cpp
  src/vk/vulkan_command_context.cpp
  src/vk/vulkan_main_context.cpp
  src/event_handler.cpp
  src/ray_marcher.cpp
  src/ui.cpp
  src/window.cpp
  src/work_context.cpp
  src/camera.cpp
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/imgui.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/backends/imgui_impl_sdl2.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/backends/imgui_impl_vulkan.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/imgui_draw.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/imgui_widgets.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/imgui_tables.cpp")

find_package(Vulkan REQUIRED)

add_executable(AutoTF_PH src/main.cpp ${SOURCE_FILES})
target_include_directories(AutoTF_PH PRIVATE "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/dependencies/"
  "${Vulkan_INCLUDE_DIRS}"
  "${SDL2_DIR}/include/"
  "${PROJECT_SOURCE_DIR}/dependencies/VulkanMemoryAllocator/include/"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/")
add_subdirectory("${SDL2_DIR}")
target_link_libraries(AutoTF_PH SDL2 ${Vulkan_LIBRARIES})
