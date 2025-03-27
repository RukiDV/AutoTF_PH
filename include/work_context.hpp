#pragma once

#include <vector>
#include "app_state.hpp"
#include "persistence.hpp"
#include "threshold_cut.hpp"
#include "ray_marcher.hpp"
#include "transfer_function.hpp"
#include "ui.hpp"
#include "vk/storage.hpp"
#include "vk/swapchain.hpp"
#include "vk/vulkan_command_context.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/renderer.hpp"
#include "vk/synchronization.hpp"
#include "volume.hpp"
#include "gpu_renderer.hpp"

namespace ve {

// manages the orchestration of rendering and computing
class WorkContext
{
public:
  WorkContext(const VulkanMainContext& vmc, VulkanCommandContext& vcc);
  void construct(AppState& app_state, const Volume& volume);
  void destruct();
  void reload_shaders();
  void draw_frame(AppState& app_state);
  vk::Extent2D recreate_swapchain(bool vsync);
  std::vector<float> get_result_values();
  void set_persistence_pairs(const std::vector<PersistencePair>& pairs, const Volume& volume);
  void set_transfer_function(const TransferFunction &tf) {
        transfer_function = tf;
    }

  // update the transfer function from a histogram
  void update_histogram_tf(const Volume &volume)
  {
      std::vector<glm::vec4> tf_data;
      // call the transfer function's histogram update
      transfer_function.update_from_histogram(volume, tf_data);
      // update the GPU buffer for the transfer function
      storage.get_buffer_by_name("transfer_function").update_data_bytes(tf_data.data(), sizeof(glm::vec4) * tf_data.size());
      vmc.logical_device.get().waitIdle();
  }

  // hybrid approach: first histogram-based TF, then refine with PH
  void update_histogram_ph_tf(const Volume &volume, int ph_threshold)
  {
      std::vector<glm::vec4> tf_data;
      // build histogram-based TF
      histogram_based_tf(volume, tf_data);

      // refine using persistent homology
      refine_with_ph(volume, ph_threshold, tf_data);

      // upload to GPU
      storage.get_buffer_by_name("transfer_function").update_data_bytes(tf_data.data(), sizeof(glm::vec4) * tf_data.size());
      vmc.logical_device.get().waitIdle();
  }

private:

const VulkanMainContext& vmc;
VulkanCommandContext& vcc;
Storage storage;
Swapchain swapchain;
Renderer renderer;
RayMarcher ray_marcher;
uint32_t read_only_buffer_idx = 0;
UI ui;
TransferFunction transfer_function;
MergeTree merge_tree;
std::vector<PersistencePair> persistence_pairs;
std::vector<Synchronization> syncs;
bool compute_finished = false;
uint32_t uniform_buffer;
void render(uint32_t image_idx, AppState& app_state, uint32_t read_only_image);
void histogram_based_tf(const Volume &volume, std::vector<glm::vec4> &tf_data); 
void refine_with_ph(const Volume &volume, int ph_threshold, std::vector<glm::vec4> &tf_data);
};
} // namespace ve
