cmake_minimum_required(VERSION 3.19)
project(AutoTF_PH)
set(CMAKE_CXX_STANDARD 20)

set(SDL3_DIR "${PROJECT_SOURCE_DIR}/dependencies/SDL3")

set(SOURCE_FILES
  src/gpu_renderer.cpp
  src/persistence.cpp
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
  src/transfer_function.cpp
  src/merge_tree.cpp
  src/threshold_cut.cpp
  src/util/texture_loader.cpp
  src/vk/device_timer.cpp
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/imgui.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/backends/imgui_impl_sdl3.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/backends/imgui_impl_vulkan.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/imgui_draw.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/imgui_widgets.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/imgui_tables.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/implot/implot.cpp"
  "${PROJECT_SOURCE_DIR}/dependencies/implot/implot_items.cpp")

find_package(Vulkan REQUIRED)
find_package(ZLIB REQUIRED)
find_package(VTK REQUIRED COMPONENTS CommonColor CommonCore RenderingCore RenderingOpenGL2 InteractionStyle FiltersCore CommonDataModel CommonExecutionModel)

add_executable(AutoTF_PH src/main.cpp ${SOURCE_FILES})
target_include_directories(AutoTF_PH PRIVATE "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/dependencies/"
  "${PROJECT_SOURCE_DIR}/dependencies/implot"
  "${Vulkan_INCLUDE_DIRS}"
  "${SDL3_DIR}/include/"
  "${PROJECT_SOURCE_DIR}/dependencies/VulkanMemoryAllocator/include/"
  "${PROJECT_SOURCE_DIR}/dependencies/imgui/"
  "${ZLIB_INCLUDE_DIRS}")
add_subdirectory("${SDL3_DIR}")

target_link_libraries(AutoTF_PH PRIVATE SDL3 ${ZLIB_LIBRARIES} ${Vulkan_LIBRARIES} ${VTK_LIBRARIES})
